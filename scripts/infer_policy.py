"""Apply a frozen prompting policy with DermSAM to a split and evaluate at original resolution.

Inference uses only the image and SAM's own predictions (no GT): round 0 = learned automatic token,
later rounds / rules / post-processing as specified by the policy JSON.

python scripts/infer_policy.py --ckpt runs/sam/<run>/best.pt --policy <policy.json | one_shot | fixed2> \
       --splits dev test ph2 --name <tag>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dermrsi.data import load_split  # noqa: E402
from dermrsi.evaluate import evaluate_probs, fmt  # noqa: E402
from dermrsi.rsi.descriptors import image_descriptors, lab_images  # noqa: E402
from dermrsi.rsi.fitness import masks_from, upsample  # noqa: E402
from dermrsi.rsi.policy import Executor, base_view, default_policy, load_coarse, validate, view_expert  # noqa: E402,E501
from dermrsi.sam_model import DermSAM  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rsi_search import encode_split_view  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRESETS = {
    # one automatic SAM pass (learned prompt token), no refinement
    "one_shot": lambda: default_policy(0),
    # one SAM pass prompted by the coarse-localiser box
    "coarse_box": lambda: default_policy(0, init="coarse"),
    # hand-designed fixed refinement: 2 rounds of box(pred, +5%) + previous mask prompt
    "fixed2": lambda: default_policy(2),
    # fixed refinement with clicks: box + mask + 1 deepest positive click + 1 background negative click
    "fixed2_clicks": lambda: {**default_policy(2), "rounds": [
        dict(default_policy(1)["rounds"][0], pos=1, neg=1) for _ in range(2)]},
}


def load_policy(spec):
    if spec in PRESETS:
        return validate(PRESETS[spec]())
    with open(spec) as f:
        return validate(json.load(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="large")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--views", nargs="+", default=None, help="override policy views")
    ap.add_argument("--splits", nargs="+", default=["dev", "test", "ph2"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--coarse", default="unet_s512")
    ap.add_argument("--experts", nargs="*", default=[], help="LETTER=variant:ckpt (see rsi_search.py)")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--save_prob", type=int, default=0,
                    help="also save the policy's continuous lesion probability (before the post-processing "
                         "threshold) as prob_<split>.npy (uint8, 512x512) -- used for pixel-level ROC curves")
    args = ap.parse_args()

    policy = load_policy(args.policy)
    if args.views:
        policy["views"] = args.views
    out_dir = os.path.join(ROOT, "runs", "final", args.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "policy.json"), "w") as f:
        json.dump(policy, f, indent=1)
    model = DermSAM.from_checkpoint(args.variant, args.ckpt)
    # 'frozen' = original pretrained weights (zero-shot expert)
    expert = {}
    for spec in args.experts:
        k, rest = spec.split("=", 1)
        var, ck = rest.split(":", 1)
        expert[k] = DermSAM.from_checkpoint(var, ck)
    results = {}
    for split in args.splits:
        t0 = time.time()
        img, _, meta = load_split(split, 512)
        lab, idesc = lab_images(img), image_descriptors(img, meta)
        coarse = load_coarse(split, args.coarse) if args.coarse else None
        masks, probs = [], []
        # process the split in chunks so that cached features of all views fit in GPU memory
        for c0 in range(0, len(meta), args.chunk):
            sub = range(c0, min(len(meta), c0 + args.chunk))
            feats = {v: [t.cuda() for t in encode_split_view(expert[view_expert(v)] if view_expert(v) else model,
                                                             split, base_view(v), subset=sub)]
                     for v in policy["views"]}
            ex = Executor(model, feats, lab[c0:sub.stop], {k: v[c0:sub.stop] for k, v in idesc.items()},
                          coarse=coarse[c0:sub.stop] if coarse is not None else None, expert=expert)
            P, post, desc, _ = ex.run(policy)
            masks += masks_from(P, post, size=512)
            if args.save_prob:
                probs.append((upsample(P, 512) * 255).round().clip(0, 255).astype(np.uint8))
            del feats, ex
            torch.cuda.empty_cache()
        u8 = (np.stack(masks).astype(np.uint8) * 255)
        np.save(os.path.join(out_dir, f"pred_{split}.npy"), u8)
        if args.save_prob:
            np.save(os.path.join(out_dir, f"prob_{split}.npy"), np.concatenate(probs))
        s, per = evaluate_probs(u8, split)
        results[split] = s
        print(args.name, split, fmt(s), f"({time.time()-t0:.0f}s)", flush=True)
        with open(os.path.join(out_dir, f"per_image_{split}.json"), "w") as f:
            json.dump(per, f)
    res_path = os.path.join(out_dir, "results.json")
    if os.path.exists(res_path):  # merge with previously evaluated splits of the same run
        results = {**json.load(open(res_path)).get("results", {}), **results}
    with open(res_path, "w") as f:
        json.dump({"ckpt": args.ckpt, "policy": policy, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
