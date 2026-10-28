"""Segmentation metrics evaluated against original-resolution ground truth.

Per-image Dice / IoU are averaged over images (primary numbers). Dataset-level (pixel-pooled)
Dice / IoU are also returned because several lightweight-segmentation papers report them.
Boundary F1 follows the DAVIS F-measure: boundary tolerance = ceil(0.008 * image diagonal).
"""
import math

import cv2
import numpy as np


def _boundary(mask):
    """One-pixel-wide boundary of a binary mask (inner contour)."""
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=bool)
    er = cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (m - er).astype(bool)


def boundary_f1(pred, gt, bound_th=0.008):
    h, w = gt.shape
    tol = max(1, math.ceil(bound_th * math.hypot(h, w)))
    bp, bg = _boundary(pred), _boundary(gt)
    if bp.sum() == 0 and bg.sum() == 0:
        return 1.0
    if bp.sum() == 0 or bg.sum() == 0:
        return 0.0
    # distance from every pixel to the nearest boundary pixel of the other mask
    dist_to_gt = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float((dist_to_gt[bp] <= tol).mean())
    recall = float((dist_to_pred[bg] <= tol).mean())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def _surface(mask):
    """Surface pixels as in medpy: foreground pixels with a 4-connected background neighbour."""
    m = mask.astype(np.uint8)
    er = cv2.erode(m, _CROSS, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (m - er).astype(bool)


def surface_distances(pred, gt):
    """HD95 and ASSD in pixels (medpy definitions, exact Euclidean distance transform).

    hd95 = 95th percentile of the pooled directed surface distances (pred->gt and gt->pred)
    assd = mean of the two average directed surface distances
    If exactly one of the masks is empty the distances are undefined; we return the image diagonal
    (worst case) and flag it. Both empty -> 0.
    """
    h, w = gt.shape
    diag = math.hypot(h, w)
    sp, sg = _surface(pred), _surface(gt)
    if not sp.any() and not sg.any():
        return 0.0, 0.0, False
    if not sp.any() or not sg.any():
        return diag, diag, True
    d_to_g = cv2.distanceTransform((~sg).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_to_p = cv2.distanceTransform((~sp).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    a, b = d_to_g[sp], d_to_p[sg]
    hd95 = float(np.percentile(np.hstack([a, b]), 95))
    assd = float((a.mean() + b.mean()) / 2)
    return hd95, assd, False


def binary_stats(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(pred.size - tp - fp - fn)
    return tp, fp, fn, tn


def image_metrics(pred, gt, with_boundary=True, with_surface=False):
    tp, fp, fn, tn = binary_stats(pred, gt)
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    out = {
        "dice": dice,
        "iou": iou,
        "sens": tp / (tp + fn) if tp + fn > 0 else 1.0,
        "spec": tn / (tn + fp) if tn + fp > 0 else 1.0,
        "acc": (tp + tn) / max(1, tp + fp + fn + tn),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    if with_boundary:
        out["bf1"] = boundary_f1(pred, gt)
    if with_surface:
        hd95, assd, empty = surface_distances(pred, gt)
        diag = math.hypot(*gt.shape)
        # px at original resolution, and normalised by the image diagonal (%; resolution independent)
        out.update(hd95=hd95, assd=assd, hd95_n=100 * hd95 / diag, assd_n=100 * assd / diag,
                   surf_empty=empty)
    return out


def summarize(per_image):
    keys = ["dice", "iou", "bf1", "sens", "spec", "acc"]
    s = {k: float(np.mean([m[k] for m in per_image])) for k in keys if k in per_image[0]}
    tp = sum(m["tp"] for m in per_image)
    fp = sum(m["fp"] for m in per_image)
    fn = sum(m["fn"] for m in per_image)
    s["dice_global"] = 2 * tp / (2 * tp + fp + fn)
    s["iou_global"] = tp / (tp + fp + fn)
    # ISIC 2018 challenge "thresholded Jaccard": per-image IoU below 0.65 counts as 0
    s["iou_thr65"] = float(np.mean([m["iou"] if m["iou"] >= 0.65 else 0.0 for m in per_image]))
    for k in ["hd95", "assd", "hd95_n", "assd_n"]:
        if k in per_image[0]:
            v = np.array([m[k] for m in per_image])
            s[k] = float(v.mean())
            s[k + "_median"] = float(np.median(v))
    if "surf_empty" in per_image[0]:
        s["surf_empty"] = int(sum(m["surf_empty"] for m in per_image))
    s["n"] = len(per_image)
    return s


def soft_dice_iou(prob, gt, thr=0.5):
    """Fast metrics on low-resolution arrays (used during training / RSI search)."""
    p = prob >= thr
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum(axis=(-1, -2))
    ps = p.sum(axis=(-1, -2))
    gs = g.sum(axis=(-1, -2))
    union = ps + gs - inter
    dice = np.where(ps + gs > 0, 2 * inter / np.maximum(ps + gs, 1), 1.0)
    iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return dice, iou
