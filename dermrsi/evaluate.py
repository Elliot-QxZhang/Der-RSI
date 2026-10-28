"""Unified evaluation: low-resolution probability maps -> original-resolution metrics.

Every method in this repo saves its predictions as an (N, S, S) uint8 probability array
(prob * 255) aligned with data/cache/<split>.json. Predictions are bilinearly resized to the
original image size, thresholded at 0.5 and compared to the original ground-truth PNG/BMP.
"""
import argparse
import json
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image

from dermrsi.data import CACHE
from dermrsi.metrics import image_metrics, summarize


_SHARED = {}
GT_CACHE = os.path.join(CACHE, "gt_full")


def _load_gt(item):
    """Original-resolution binary GT, cached as packed bits (reading 3000x2000 PNGs dominates eval time)."""
    f = os.path.join(GT_CACHE, item["id"] + ".npz")
    if os.path.exists(f):
        z = np.load(f)
        return np.unpackbits(z["b"])[: z["h"] * z["w"]].reshape(z["h"], z["w"]).astype(bool)
    gt = np.asarray(Image.open(item["gt"]).convert("L")) >= 128
    os.makedirs(GT_CACHE, exist_ok=True)
    tmp = f + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, b=np.packbits(gt), h=gt.shape[0], w=gt.shape[1])
    os.replace(tmp, f)
    return gt


def _init_worker(path, surface=False):
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    _SHARED["probs"] = np.load(path, mmap_mode="r")
    _SHARED["surface"] = surface


def _eval_one(args):
    i, item, thr = args
    prob = np.asarray(_SHARED["probs"][i])
    gt = _load_gt(item)
    h, w = gt.shape
    p = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if prob.dtype == np.uint8:
        p = p / 255.0
    pred = p >= thr
    m = image_metrics(pred, gt, with_surface=_SHARED.get("surface", False))
    m["id"] = item["id"]
    return m


def evaluate_probs(probs, split, thr=0.5, workers=24, surface=False):
    """probs: (N,S,S) array (uint8 = prob*255, or float in [0,1]) or path to such an .npy file.
    Workers memory-map the predictions instead of receiving them through pickling.
    surface=True additionally computes HD95 / ASSD (pixels at original resolution, medpy definitions)."""
    with open(_cache_json(split)) as f:
        meta = json.load(f)
    tmp = None
    if isinstance(probs, str):
        path = probs
    else:
        arr = np.asarray(probs)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
        fd, tmp = tempfile.mkstemp(suffix=".npy", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
        os.close(fd)
        np.save(tmp, arr)
        path = tmp
    n = np.load(path, mmap_mode="r").shape[0]
    assert len(meta) == n, (len(meta), n)
    jobs = [(i, meta[i], thr) for i in range(n)]
    try:
        # spawn (not fork): forking a process that holds CUDA / OpenMP state can deadlock the workers
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker, initargs=(path, surface)) as ex:
            per_image = list(ex.map(_eval_one, jobs, chunksize=8))
    finally:
        if tmp:
            os.remove(tmp)
    return summarize(per_image), per_image


def _cache_json(split):
    return os.path.join(CACHE, f"{split}.json")


def fmt(s):
    return (f"Dice {s['dice']*100:.2f} | IoU {s['iou']*100:.2f} | BF1 {s['bf1']*100:.2f} | "
            f"gDice {s['dice_global']*100:.2f} | gIoU {s['iou_global']*100:.2f} | "
            f"thrJ {s['iou_thr65']*100:.2f} | Sens {s['sens']*100:.2f} | Spec {s['spec']*100:.2f} | "
            f"Acc {s['acc']*100:.2f} | n={s['n']}" +
            (f" | HD95 {s['hd95']:.1f}px ({s['hd95_n']:.2f}%diag) | ASSD {s['assd']:.1f}px ({s['assd_n']:.2f}%diag)"
             if "hd95" in s else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", help="npy file with (N,S,S) uint8 probabilities")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default=None, help="json file for summary + per-image metrics")
    ap.add_argument("--surface", type=int, default=1, help="also compute HD95 / ASSD")
    args = ap.parse_args()
    s, per = evaluate_probs(args.pred, args.split, surface=bool(args.surface))
    print(args.pred, args.split, fmt(s))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": s, "per_image": per}, f)


if __name__ == "__main__":
    main()
