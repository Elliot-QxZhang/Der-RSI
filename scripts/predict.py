"""End-to-end Derm-RSI-SAM inference on raw dermoscopy images (no ground truth used).

  coarse U-Net (ResNet34, 512px, flip-TTA)  ->  SAM-family segmenter with frozen weights
      default: MedSAM ViT-B + learned prompts (8 context tokens, lesion token, dense prompt bias)
      --backbone sam2L: SAM 2.1 Hiera-L + LoRA + learned lesion token
  -> frozen RSI policy (GT-free rules over dermoscopic descriptors)  ->  binary masks at original size

python scripts/predict.py --images '/path/to/*.jpg' --out preds/            # writes <name>_mask.png
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dermrsi.baselines import build  # noqa: E402
from dermrsi.data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from dermrsi.rsi.descriptors import image_descriptors, lab_images  # noqa: E402
from dermrsi.rsi.fitness import masks_from  # noqa: E402
from dermrsi.rsi.policy import Executor, base_view, validate, view_tensor  # noqa: E402
from dermrsi.sam_model import DermSAM  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(ROOT, "checkpoints", "final")


def norm(img, size):
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC)
    return torch.from_numpy(((x.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))


@torch.no_grad()
def run_batch(paths, unet, sam, policy, loader=None):
    """paths: image files; loader(path) -> RGB uint8 array (default: PIL decode)."""
    imgs = [loader(p) if loader else np.asarray(Image.open(p).convert("RGB")) for p in paths]
    meta = [{"h": im.shape[0], "w": im.shape[1]} for im in imgs]
    im512 = np.stack([cv2.resize(im, (512, 512), interpolation=cv2.INTER_CUBIC) for im in imgs])
    # coarse localiser (flip TTA as in training-time evaluation)
    x = torch.stack([norm(im, 512) for im in imgs]).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = torch.sigmoid(unet(x).float())
        p = (p + torch.sigmoid(unet(x.flip(-1)).float()).flip(-1) + torch.sigmoid(unet(x.flip(-2)).float()).flip(-2)) / 3
    coarse = F.interpolate(p, size=(256, 256), mode="area")[:, 0]
    # SAM features for every view of the policy
    x = torch.stack([norm(im, 1024) for im in imgs]).cuda()
    feats = {}
    for v in policy["views"]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats[v] = [f.to(torch.bfloat16) for f in sam.encode(view_tensor(x, base_view(v)))]
    ex = Executor(sam, feats, lab_images(im512), image_descriptors(im512, meta), coarse=coarse)
    P, post, _, _ = ex.run(policy, bs=len(paths))
    masks = masks_from(P, post, size=512)
    return [cv2.resize(m.astype(np.uint8), (mt["w"], mt["h"]), interpolation=cv2.INTER_LINEAR) > 0
            for m, mt in zip(masks, meta)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="image files or glob patterns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="medsamP", choices=["medsamP", "sam2L"])
    ap.add_argument("--sam_ckpt", default=None)
    ap.add_argument("--unet_ckpt", default=os.path.join(FINAL, "coarse_unet_r34_s512.pt"))
    ap.add_argument("--policy", default=None)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    paths = sorted(p for pat in args.images for p in glob.glob(pat))
    os.makedirs(args.out, exist_ok=True)
    unet = build("unet").cuda().eval()
    unet.load_state_dict(torch.load(args.unet_ckpt, map_location="cuda"))
    variant, ck, pol = {"medsamP": ("medsam", "medsam_prompt8.pt", "rsi_policy_medsamP.json"),
                        "sam2L": ("large", "dermsam2L_lora_r16.pt", "rsi_policy_sam2L.json")}[args.backbone]
    sam = DermSAM.from_checkpoint(variant, args.sam_ckpt or os.path.join(FINAL, ck))
    policy = validate(json.load(open(args.policy or os.path.join(FINAL, pol))))
    for s in range(0, len(paths), args.bs):
        chunk = paths[s:s + args.bs]
        for p, m in zip(chunk, run_batch(chunk, unet, sam, policy)):
            name = os.path.splitext(os.path.basename(p))[0]
            Image.fromarray(m.astype(np.uint8) * 255).save(os.path.join(args.out, f"{name}_mask.png"))
        print(f"{min(len(paths), s + args.bs)}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
