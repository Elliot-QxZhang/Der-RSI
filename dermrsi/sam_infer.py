"""Inference helpers for DermSAM: cached image features + batched multi-round decoding."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dermrsi.data import SegDataset, load_split
from dermrsi.metrics import soft_dice_iou
from dermrsi.prompts import down, mask_box, scale_box


@torch.no_grad()
def encode_split(model, split, bs=8, flip=False, device="cuda"):
    """Returns list of 3 CPU bf16 tensors (N,C,H,W) with SAM image features."""
    ds = SegDataset(split, 1024, out_size=1024, train=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8)
    out = None
    model.eval()
    for x, _, _ in dl:
        x = x.to(device, non_blocking=True)
        if flip:
            x = x.flip(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.encode(x)
        if out is None:
            out = [[] for _ in f]
        for i in range(len(f)):
            out[i].append(f[i].to(torch.bfloat16).cpu())
    return [torch.cat(o) for o in out]


def batch_feats(feats, idx, device="cuda"):
    return [f[idx].to(device, non_blocking=True) for f in feats]


@torch.no_grad()
def simple_rounds(model, feats, n_rounds=2, bs=16, expand=0.05, use_mask=True):
    """Fixed (non-learned) refinement: auto -> box(pred)+mask prompt, repeated."""
    n = feats[0].shape[0]
    outs = [[] for _ in range(n_rounds)]
    for s in range(0, n, bs):
        idx = torch.arange(s, min(n, s + bs))
        f = batch_feats(feats, idx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _ = model.decode(f)
            lg = lg.float()
            outs[0].append(lg.cpu())
            for r in range(1, n_rounds):
                box = scale_box(mask_box(down(lg, 256)[:, 0] > 0), expand)
                lg, _ = model.decode(f, boxes=box, mask_logits=lg if use_mask else None)
                lg = lg.float()
                outs[r].append(lg.cpu())
    return [torch.cat(o) for o in outs]


def logits_to_prob512(lg):
    """(N,1,256,256) or (N,256,256) logits -> (N,512,512) float32 probabilities (numpy)."""
    if lg.dim() == 3:
        lg = lg[:, None]
    out = []
    for s in range(0, lg.shape[0], 64):
        x = F.interpolate(lg[s:s + 64].float(), size=(512, 512), mode="bilinear", align_corners=False)
        out.append(torch.sigmoid(x)[:, 0].numpy())
    return np.concatenate(out)


@torch.no_grad()
def evaluate_dev(model, split="dev", n_rounds=2):
    feats = encode_split(model, split)
    _, gt, _ = load_split(split, 512)
    gt = np.asarray(gt)
    lgs = simple_rounds(model, feats, n_rounds=n_rounds)
    res = {}
    for r, lg in enumerate(lgs):
        d, j = soft_dice_iou(logits_to_prob512(lg), gt)
        res[f"r{r}_dice"] = float(d.mean())
        res[f"r{r}_iou"] = float(j.mean())
    del feats
    torch.cuda.empty_cache()
    return res
