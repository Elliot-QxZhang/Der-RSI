"""Dermoscopic descriptors used by the RSI policy (GT-free) and by failure analysis.

Image-level (precomputed from the image only):
  hair      fraction of pixels with strong black-hat response (dark thin hair / ruler lines)
  vignette  brightness of the four image corners relative to the centre (dark circular field of view)
  lum       mean L* of the image
  res       log10 of the original image size in megapixels (acquisition device / magnification proxy)
Prediction-level (computed on GPU from the current consensus probability map P, 256x256):
  area      predicted lesion area fraction
  border    fraction of image-border pixels covered by the prediction (lesion larger than field of view)
  unc       uncertain-band area (0.25<P<0.75) relative to predicted area (fuzzy / low-contrast edge)
  contrast  Lab colour distance between an inner and an outer ring around the predicted boundary
  compact   isoperimetric ratio perimeter^2 / (4*pi*area) (border irregularity / fragmentation)
  piou      SAM IoU-head estimate (mean over TTA views)
  tta_dis   1 - mean IoU between TTA-view masks (prediction instability)
  coarse_iou IoU between the SAM prediction and the coarse localiser mask (cross-model agreement)
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

FEATURES = ["hair", "vignette", "lum", "res", "area", "border", "unc", "contrast", "compact", "piou", "tta_dis",
            "coarse_iou"]


def _image_desc(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    hair = float((bh > 25).mean())
    h, w = g.shape
    c = max(4, h // 10)
    corners = np.concatenate([g[:c, :c].ravel(), g[:c, -c:].ravel(), g[-c:, :c].ravel(), g[-c:, -c:].ravel()])
    centre = g[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    vignette = float(corners.mean() / max(1.0, centre.mean()))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return hair, vignette, float(lab[..., 0].mean()) / 255 * 100


def image_descriptors(images, meta=None):
    """images: (N,H,W,3) uint8 (any size; downsampled to 256). Returns dict of (N,) float arrays."""
    out = {"hair": [], "vignette": [], "lum": []}
    if meta is not None:
        out["res"] = [np.log10(m["h"] * m["w"] / 1e6) for m in meta]
    for im in images:
        im = cv2.resize(np.asarray(im), (256, 256), interpolation=cv2.INTER_AREA)
        h, v, l = _image_desc(im)
        out["hair"].append(h)
        out["vignette"].append(v)
        out["lum"].append(l)
    return {k: np.array(v, np.float32) for k, v in out.items()}


def lab_images(images):
    """(N,H,W,3) uint8 -> (N,3,256,256) float16 Lab tensor (L in [0,100], a/b centred)."""
    out = np.zeros((len(images), 3, 256, 256), np.float16)
    for i, im in enumerate(images):
        im = cv2.resize(np.asarray(im), (256, 256), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(im.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        out[i] = lab.transpose(2, 0, 1)
    return torch.from_numpy(out)


def _dilate(m, r):
    return F.max_pool2d(m, 2 * r + 1, 1, r) if r > 0 else m


def _erode(m, r):
    return -F.max_pool2d(-m, 2 * r + 1, 1, r) if r > 0 else m


@torch.no_grad()
def pred_descriptors(P, lab, piou, view_masks):
    """P (B,256,256) consensus prob, lab (B,3,256,256), piou (B,), view_masks (V,B,256,256) bool."""
    m = (P > 0.5).float()[:, None]
    area_px = m.flatten(1).sum(1)
    area = area_px / m[0].numel()
    bd = torch.zeros_like(m[0, 0], dtype=torch.bool)
    bd[0, :] = bd[-1, :] = bd[:, 0] = bd[:, -1] = True
    border = (m[:, 0] * bd).flatten(1).sum(1) / bd.sum()
    unc = ((P > 0.25) & (P < 0.75)).flatten(1).sum(1).float() / area_px.clamp(min=1)
    inner = (m - _erode(m, 6)).clamp(0, 1) * _erode(m, 2)
    outer = (_dilate(m, 8) - _dilate(m, 3)).clamp(0, 1)
    lab = lab.float()

    def ring_mean(r):
        s = r.flatten(1).sum(1).clamp(min=1)
        return (lab * r).flatten(2).sum(2) / s[:, None]

    contrast = (ring_mean(inner) - ring_mean(outer)).norm(dim=1)
    perim = (m - _erode(m, 1)).clamp(0, 1).flatten(1).sum(1)
    compact = perim ** 2 / (4 * np.pi * area_px.clamp(min=1))
    v = view_masks.float()
    if v.shape[0] > 1:
        ious = []
        for i in range(v.shape[0]):
            for j in range(i + 1, v.shape[0]):
                inter = (v[i] * v[j]).flatten(1).sum(1)
                uni = ((v[i] + v[j]) > 0).float().flatten(1).sum(1)
                ious.append(torch.where(uni > 0, inter / uni.clamp(min=1), torch.ones_like(uni)))
        tta_dis = 1 - torch.stack(ious).mean(0)
    else:
        tta_dis = torch.zeros_like(area)
    return {"area": area, "border": border, "unc": unc, "contrast": contrast, "compact": compact,
            "piou": piou.float(), "tta_dis": tta_dis}
