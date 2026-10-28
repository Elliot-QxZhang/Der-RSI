"""SAM 2.1 (Hiera) with low-rank adapters and a learnable automatic prompt token.

The pretrained SAM 2.1 weights are kept frozen; we train
  * LoRA adapters on every attention qkv / proj and MLP linear of the Hiera image encoder,
  * the FPN neck, prompt encoder and mask decoder (small: ~4M params),
  * a learnable "lesion" prompt token used when no geometric prompt is available (round 0).

`DermSAM.encode(images)` computes image features once; `DermSAM.decode(feats, prompts)` can then be
called repeatedly with different prompts (box / points / mask), which is what the RSI policy does.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

HF_IDS = {
    "large": "facebook/sam2.1-hiera-large",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    # MedSAM ViT-B (SAM fine-tuned on 1.5M medical image-mask pairs); local safetensors conversion of
    # flaviagiammarino/medsam-vit-base (= official medsam_vit_b.pth)
    "medsam": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints",
                           "medsam-vit-base-fg"),
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale


def add_lora(module, r, alpha, names=("qkv", "proj", "proj_in", "proj_out", "lin1", "lin2")):
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in names:
            setattr(module, name, LoRALinear(child, r, alpha))
            n += 1
        else:
            n += add_lora(child, r, alpha, names)
    return n


class DermSAM(nn.Module):
    def __init__(self, variant="large", lora_r=16, lora_alpha=16, grad_ckpt=True, tune="lora", n_ctx=4):
        """tune = "lora"  : LoRA on the image encoder + trainable neck / prompt encoder / mask decoder
               "prompt": prompt learning only -- every pretrained weight frozen; learnable lesion token
                         (round-0 prompt) + n_ctx context tokens appended to every prompt + a dense
                         prompt bias. Context tokens start as SAM's "not-a-point" padding embedding and the
                         dense bias at zero, so the untrained model is exactly the zero-shot model."""
        super().__init__()
        self.variant = variant
        self.tune = tune
        self.v1 = variant == "medsam"  # SAM-v1 architecture (single-scale ViT features)
        if self.v1:
            from transformers import SamModel
            self.sam = SamModel.from_pretrained(HF_IDS[variant], attn_implementation="sdpa")
        else:
            from transformers import Sam2Model
            self.sam = Sam2Model.from_pretrained(HF_IDS[variant], attn_implementation="sdpa")
            self.sam.mask_decoder.dynamic_multimask_via_stability = False
        for p in self.sam.parameters():
            p.requires_grad = False
        enc = self.sam.vision_encoder
        blocks = enc.layers if self.v1 else enc.backbone.blocks
        n = 0
        if tune == "lora":
            n = add_lora(blocks, lora_r, lora_alpha)
            # trainable: neck, prompt encoder (except the fixed random-Fourier buffer), mask decoder
            for m in [enc.neck, self.sam.prompt_encoder, self.sam.mask_decoder]:
                for p in m.parameters():
                    p.requires_grad = True
            if not self.v1:
                self.sam.no_memory_embedding.requires_grad = True
        dim0 = self.sam.config.prompt_encoder_config.hidden_size
        self.n_ctx = n_ctx if tune == "prompt" else 0
        if self.n_ctx:
            nap = self.sam.prompt_encoder.not_a_point_embed.weight.detach().clone()
            self.ctx_tokens = nn.Parameter(nap.view(1, 1, 1, dim0).repeat(1, 1, self.n_ctx, 1))
            hw = self.sam.prompt_encoder.image_embedding_size
            self.dense_bias = nn.Parameter(torch.zeros(1, dim0, *hw))
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        dim = self.sam.config.prompt_encoder_config.hidden_size
        self.auto_token = nn.Parameter(torch.randn(1, 1, 2, dim) * 0.02)
        self.n_lora = n
        self.image_size = self.sam.config.prompt_encoder_config.image_size  # 1024
        if grad_ckpt and tune == "lora":
            from functools import partial
            import torch.utils.checkpoint as ckpt
            for blk in blocks:
                blk.gradient_checkpointing = True
                blk._gradient_checkpointing_func = partial(ckpt.checkpoint, use_reentrant=False)

    # ---------------------------------------------------------------- image features
    def encode(self, pixel_values):
        """pixel_values: ImageNet-normalised (B,3,1024,1024). Returns a list of feature maps (last = main)."""
        if self.tune == "prompt":  # frozen encoder: no activations kept for backward
            with torch.no_grad():
                return [f.detach() for f in self._encode(pixel_values)]
        return self._encode(pixel_values)

    def _encode(self, pixel_values):
        sam = self.sam
        b = pixel_values.shape[0]
        if self.v1:
            # MedSAM preprocessing: per-image min-max scaling of the raw image to [0,1]
            x = pixel_values * self._std + self._mean
            lo = x.flatten(1).min(1).values.view(b, 1, 1, 1)
            hi = x.flatten(1).max(1).values.view(b, 1, 1, 1)
            x = (x - lo) / (hi - lo).clamp(min=1e-6)
            return [sam.vision_encoder(x)[0]]
        feats, _, _, _ = sam.get_image_features(pixel_values)
        feats[-1] = feats[-1] + sam.no_memory_embedding
        return [f.permute(1, 2, 0).reshape(b, -1, *s) for f, s in zip(feats, sam.backbone_feature_sizes)]

    # ---------------------------------------------------------------- prompted decoding
    def decode(self, feats, boxes=None, points=None, labels=None, mask_logits=None, use_auto=None, mask_on=None, box_on=None):
        """
        boxes : (B,4) xyxy in 1024 pixel space or None
        points: (B,K,2) xy in 1024 space, labels (B,K) in {1,0,-10(pad)} or None
        mask_logits: (B,1,256,256) dense prompt (previous-round logits) or None
        use_auto: (B,) bool tensor, whether to prepend the learned auto token (default: True when no
                  box and no points are given)
        returns logits (B,1,256,256), iou_pred (B,)
        """
        sam = self.sam
        pe = sam.prompt_encoder
        b = feats[-1].shape[0]
        dev = feats[-1].device
        sparse = []
        if points is not None:
            pts = points[:, None].float()
            lab = labels[:, None].long()
            sparse.append(pe._embed_points(pts, lab, pad=boxes is None))
        if boxes is not None:
            be = pe._embed_boxes(boxes[:, None].float().clone())
            if box_on is not None:  # per-image: disabled boxes become "not-a-point" tokens
                nap_b = pe.not_a_point_embed.weight.view(1, 1, 1, -1).expand_as(be)
                be = torch.where(box_on.view(b, 1, 1, 1), be, nap_b)
            sparse.append(be)
        if use_auto is None:
            use_auto = torch.full((b,), points is None and boxes is None, dtype=torch.bool, device=dev)
        tok = self.auto_token.expand(b, 1, -1, -1)
        # when not used, the auto token is replaced by "not-a-point" embeddings (no effect on SAM)
        nap = pe.not_a_point_embed.weight.view(1, 1, 1, -1).expand_as(tok)
        sparse.append(torch.where(use_auto.view(b, 1, 1, 1), tok, nap))
        if self.n_ctx:
            sparse.append(self.ctx_tokens.expand(b, 1, -1, -1))
        sparse = torch.cat(sparse, dim=2)
        no_mask = pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(b, -1, *pe.image_embedding_size)
        if mask_logits is not None:
            dense = pe.mask_embed(mask_logits.to(feats[-1].dtype))
            if mask_on is not None:
                dense = torch.where(mask_on.view(b, 1, 1, 1), dense, no_mask.to(dense.dtype))
        else:
            dense = no_mask
        if self.n_ctx:
            dense = dense + self.dense_bias.to(dense.dtype)
        img_pe = sam.get_image_wide_positional_embeddings().repeat(b, 1, 1, 1)
        if self.v1:
            low, iou = sam.mask_decoder(
                image_embeddings=feats[-1],
                image_positional_embeddings=img_pe.to(feats[-1].dtype),
                sparse_prompt_embeddings=sparse.to(feats[-1].dtype),
                dense_prompt_embeddings=dense.to(feats[-1].dtype),
                multimask_output=False,
            )
            return low[:, 0], iou[:, 0, 0]
        low, iou, _, _ = sam.mask_decoder(
            image_embeddings=feats[-1],
            image_positional_embeddings=img_pe,
            sparse_prompt_embeddings=sparse.to(feats[-1].dtype),
            dense_prompt_embeddings=dense.to(feats[-1].dtype),
            multimask_output=False,
            high_resolution_features=feats[:-1],
        )
        return low[:, 0], iou[:, 0, 0]

    def trainable_state_dict(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in keep}

    @classmethod
    def from_checkpoint(cls, variant, path, device="cuda"):
        """Build the right model (LoRA vs prompt-learning) for a trainable-state checkpoint.
        path == "frozen": the original pretrained weights (zero-shot)."""
        if path == "frozen":
            return cls(variant, grad_ckpt=False).to(device).eval()
        sd = torch.load(path, map_location=device)
        tune = "prompt" if "ctx_tokens" in sd else "lora"
        n_ctx = sd["ctx_tokens"].shape[2] if tune == "prompt" else 4
        m = cls(variant, grad_ckpt=False, tune=tune, n_ctx=n_ctx).to(device).eval()
        m.load_trainable(sd)
        return m

    def load_trainable(self, sd):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        return missing
