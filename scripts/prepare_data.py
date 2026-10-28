"""Build resized caches and split files for ISIC2018 Task 1 and PH2.

Protocol (shared by every method in this repo):
  train : official ISIC2018 Task1 training images minus a fixed 300-image hold-out (2294)
  dev   : that 300-image hold-out + the official validation set (400); used for checkpoint
          selection and RSI policy learning
  test  : official ISIC2018 Task1 test set (1000); only touched for final reporting
  ph2   : PH2 (200), external test
Images are squashed to SxS (SAM2 convention); original-resolution GT is kept for evaluation.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "cache")


def isic_items(split):
    img_dir = os.path.join(RAW, f"ISIC2018_Task1-2_{split}_Input")
    gt_dir = os.path.join(RAW, f"ISIC2018_Task1_{split}_GroundTruth")
    items = []
    for p in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        name = os.path.basename(p)[:-4]
        g = os.path.join(gt_dir, f"{name}_segmentation.png")
        assert os.path.exists(g), g
        items.append({"id": name, "img": p, "gt": g})
    return items


def ph2_items():
    items = []
    for d in sorted(glob.glob(os.path.join(RAW, "PH2Dataset", "PH2 Dataset images", "IMD*"))):
        name = os.path.basename(d)
        items.append({
            "id": name,
            "img": os.path.join(d, f"{name}_Dermoscopic_Image", f"{name}.bmp"),
            "gt": os.path.join(d, f"{name}_lesion", f"{name}_lesion.bmp"),
        })
    return items


def isic17_items(exclude_ids):
    """ISIC 2017 Part 1 test images that do NOT appear in any ISIC 2018 split used for training/selection.

    ISIC 2018 Task 1 training data contains most of ISIC 2017 (467 of the 600 ISIC 2017 test images are in our
    train split and 62 in dev), so only the remaining unseen images form a fair external set.
    """
    img_dir = os.path.join(RAW, "ISIC-2017_Test_v2_Data")
    gt_dir = os.path.join(RAW, "ISIC-2017_Test_v2_Part1_GroundTruth")
    items = []
    for p in sorted(glob.glob(os.path.join(img_dir, "ISIC_*.jpg"))):
        name = os.path.basename(p)[:-4]
        if "superpixels" in name or name in exclude_ids:
            continue
        items.append({"id": name, "img": p, "gt": os.path.join(gt_dir, f"{name}_segmentation.png")})
    return items


def load_pair(item, size):
    im = Image.open(item["img"]).convert("RGB")
    gt = Image.open(item["gt"]).convert("L")
    assert im.size == gt.size, (item, im.size, gt.size)
    w, h = im.size
    im_r = np.asarray(im.resize((size, size), Image.BICUBIC), dtype=np.uint8)
    gt_r = np.asarray(gt.resize((size, size), Image.BILINEAR), dtype=np.uint8) >= 128
    return im_r, gt_r.astype(np.uint8), (h, w)


def build(name, items, size):
    os.makedirs(OUT, exist_ok=True)
    n = len(items)
    img_mm = np.lib.format.open_memmap(os.path.join(OUT, f"{name}_img{size}.npy"), "w+", np.uint8, (n, size, size, 3))
    gt_mm = np.lib.format.open_memmap(os.path.join(OUT, f"{name}_gt{size}.npy"), "w+", np.uint8, (n, size, size))
    shapes = []
    with ProcessPoolExecutor(32) as ex:
        for i, (im, gt, hw) in enumerate(ex.map(load_pair, items, [size] * n, chunksize=8)):
            img_mm[i] = im
            gt_mm[i] = gt
            shapes.append(hw)
    img_mm.flush()
    gt_mm.flush()
    meta = [dict(it, h=hw[0], w=hw[1]) for it, hw in zip(items, shapes)]
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(meta, f)
    print(name, n, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 512])
    ap.add_argument("--n_holdout", type=int, default=300)
    ap.add_argument("--only", nargs="*", default=None, help="build only these splits (e.g. isic17)")
    args = ap.parse_args()

    train_all = isic_items("Training")
    rng = np.random.RandomState(2018)
    perm = rng.permutation(len(train_all))
    hold = set(perm[: args.n_holdout].tolist())
    train = [it for i, it in enumerate(train_all) if i not in hold]
    dev = [it for i, it in enumerate(train_all) if i in hold] + isic_items("Validation")
    splits = {"train": train, "dev": dev, "test": isic_items("Test"), "ph2": ph2_items()}
    if os.path.isdir(os.path.join(RAW, "ISIC-2017_Test_v2_Data")):
        seen = {it["id"] for it in train + dev}
        splits["isic17"] = isic17_items(seen)
    for size in args.sizes:
        for name, items in splits.items():
            if args.only and name not in args.only:
                continue
            build(name, items, size)


if __name__ == "__main__":
    main()
