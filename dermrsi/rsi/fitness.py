"""Post-processing of consensus probabilities and fast policy fitness on the dev split."""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes

from dermrsi.metrics import boundary_f1

LCC = ["none", "largest", "keep_big"]


def postprocess(prob, thr=0.5, fill=True, lcc="none", keep_frac=0.1):
    m = prob >= thr
    if lcc != "none" and m.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if lcc == "largest":
                keep = [1 + int(np.argmax(areas))]
            else:
                keep = [1 + i for i, a in enumerate(areas) if a >= keep_frac * areas.max()]
            m = np.isin(lab, keep)
    if fill:
        m = binary_fill_holes(m)
    return m


def upsample(P, size):
    """(N,256,256) torch -> (N,size,size) numpy float32."""
    out = []
    for s in range(0, P.shape[0], 64):
        x = F.interpolate(P[s:s + 64, None].float(), size=(size, size), mode="bilinear", align_corners=False)
        out.append(x[:, 0].numpy())
    return np.concatenate(out)


def _post_args(post, i):
    return dict(thr=float(post["thr"][i]), fill=bool(post["fill"][i]), lcc=LCC[int(post["lcc"][i])],
                keep_frac=float(post["keep_frac"][i]))


def masks_from(P, post, size=512, workers=32):
    probs = upsample(P, size)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda i: postprocess(probs[i], **_post_args(post, i)), range(len(probs))))


def per_image_scores(masks, gts, with_bf1=True, workers=32):
    def one(i):
        p, g = masks[i], gts[i].astype(bool)
        tp = np.logical_and(p, g).sum()
        fp = np.logical_and(p, ~g).sum()
        fn = np.logical_and(~p, g).sum()
        d = 2 * tp / max(1, 2 * tp + fp + fn) if (tp + fp + fn) else 1.0
        j = tp / max(1, tp + fp + fn) if (tp + fp + fn) else 1.0
        b = boundary_f1(p, g) if with_bf1 else 0.0
        gs = max(1, g.sum())
        return d, j, b, fp / gs, fn / gs
    with ThreadPoolExecutor(workers) as ex:
        r = np.array(list(ex.map(one, range(len(masks)))), np.float64)
    return {"dice": r[:, 0], "iou": r[:, 1], "bf1": r[:, 2], "fp_rate": r[:, 3], "fn_rate": r[:, 4]}


def fitness(scores, idx=None):
    if idx is None:
        idx = np.arange(len(scores["dice"]))
    return float((scores["dice"][idx] + scores["iou"][idx] + scores["bf1"][idx]).mean() / 3)


def summary(scores, idx=None):
    if idx is None:
        idx = np.arange(len(scores["dice"]))
    return {k: float(scores[k][idx].mean()) for k in ["dice", "iou", "bf1"]}
