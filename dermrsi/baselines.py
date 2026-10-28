"""Baseline model zoo: U-Net, UNet++, DeepLabv3+, SegFormer (smp), LB-UNet and BDFormer (official code).

Every wrapper exposes   forward(x) -> logits (B,1,H,W)   for inference and
                        loss(x, y, epoch) -> scalar       for training,
so that scripts/train_baseline.py can treat all of them identically.
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")


def bce_dice(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits).flatten(1)
    t = y.flatten(1)
    dice = 1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)
    return bce + dice.mean()


class SMPWrapper(nn.Module):
    def __init__(self, arch, encoder, weights="imagenet"):
        super().__init__()
        import segmentation_models_pytorch as smp
        cls = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus, "deeplabv3p": smp.DeepLabV3Plus,
               "segformer": smp.Segformer}[arch]
        self.net = cls(encoder_name=encoder, encoder_weights=weights, classes=1)

    def forward(self, x):
        out = self.net(x)
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out

    def loss(self, x, y, epoch):
        return bce_dice(self(x), y[:, None])


# ---------------------------------------------------------------- LB-UNet (MICCAI 2024)
def lbunet_boundary_target(mask):
    """Approximation of LB-UNet's `points_boundary2` maps (boundary.py), computed on the fly.

    Official maps = GT values on a 1-px cross-dilated inner boundary + GT values inside six radius-5
    disks centred on six contour key points chosen by a genetic algorithm that maximises the IoU of
    the hexagon they span. We take the six vertices of the polygon from cv2.approxPolyDP that are
    farthest apart along the contour (same role: key points spanning a max-IoU hexagon).
    """
    m = (mask > 0.5).astype(np.uint8)
    out = np.zeros_like(m)
    if m.sum() == 0:
        return out.astype(np.float32)
    er = cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    edge = m - er
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    out = cv2.dilate(edge, cross) * m
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).squeeze(1)
    if len(c) >= 6:
        idx = np.linspace(0, len(c), 7)[:-1].astype(int)
        best, best_iou = idx, -1
        # small random search over hexagons (stand-in for the GA)
        rng = np.random.default_rng(0)
        for _ in range(64):
            cand = np.sort(rng.choice(len(c), 6, replace=False))
            poly = np.zeros_like(m)
            cv2.fillPoly(poly, [c[cand].reshape(-1, 1, 2)], 1)
            inter = (poly & m).sum()
            iou = inter / max(1, (poly | m).sum())
            if iou > best_iou:
                best, best_iou = cand, iou
        disk = np.zeros_like(m)
        for x, y in c[best]:
            cv2.circle(disk, (int(x), int(y)), 5, 1, -1)
        out = np.maximum(out, disk * m)
    return out.astype(np.float32)


class LBUNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        sys.path.insert(0, os.path.join(TP, "LB-UNet"))
        from lbunet import LBUNet
        sys.path.pop(0)
        self.net = LBUNet(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64])

    def forward(self, x):
        _, _, out = self.net(x)
        return torch.logit(out.clamp(1e-6, 1 - 1e-6))

    @staticmethod
    def _bd(pred, t, wb, wd):
        pred, t = pred.float(), t.float()
        bce = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6).flatten(1), t.flatten(1))
        p, q = pred.flatten(1), t.flatten(1)
        dice = 1 - ((2 * (p * q).sum(1) + 1) / (p.sum(1) + q.sum(1) + 1)).mean()
        return wd * dice + wb * bce

    def loss(self, x, y, epoch):
        # GT_BceDiceLoss from the official utils.py
        pts = torch.from_numpy(np.stack([lbunet_boundary_target(m) for m in y.cpu().numpy()])).to(y.device)
        y1, pts = y[:, None], pts[:, None]
        gt_pre, key_points, out = self.net(x)
        with torch.autocast("cuda", enabled=False):
            return self._lb_loss(gt_pre, key_points, out, y1, pts)

    def _lb_loss(self, gt_pre, key_points, out, y1, pts):
        loss = self._bd(out, y1, 1, 2)
        k9, k10, k11 = key_points
        loss = loss + 0.1 * self._bd(k9, pts, 0.5, 1) + 0.2 * self._bd(k10, pts, 0.5, 1) + 0.3 * self._bd(k11, pts, 0.5, 1)
        for w, g in zip([0.1, 0.2, 0.3, 0.4, 0.5], gt_pre):
            loss = loss + w * self._bd(g, y1, 0.5, 1)
        return loss


# ---------------------------------------------------------------- BDFormer (AIIM 2025)
class BDFormerWrapper(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        root = os.path.join(TP, "BDFormer")
        sys.path.insert(0, root)
        from model.BDFormer import BDFormer
        import utils as bd_utils
        sys.path.pop(0)
        self.bd_utils = bd_utils
        self.net = BDFormer(img_size=256, in_channels=3, num_classes=1, window_size=8)
        if pretrained:
            # The official repo has the ImageNet initialisation commented out; we restore it with the
            # timm MaxViT-rmlp-small checkpoint (all 651 backbone tensors match).
            import timm
            src = timm.create_model("maxvit_rmlp_small_rw_224.sw_in1k", pretrained=True).state_dict()
            bb = self.net.multi_task_MaxViT.backbone
            missing = bb.load_state_dict(src, strict=False)
            print("BDFormer backbone init:", missing)

    def forward(self, x):
        seg, _ = self.net(x)
        return torch.logit(seg.clamp(1e-6, 1 - 1e-6))

    def loss(self, x, y, epoch):
        # contour target as in dataset/npy_datasets.py (Canny + 9x9 dilation for training)
        cont = []
        for m in y.cpu().numpy().astype(np.uint8):
            c = cv2.Canny(m, 0, 1) / 255
            cont.append(cv2.dilate(c, np.ones((9, 9), np.uint8)))
        cont = torch.from_numpy(np.stack(cont)).float().to(y.device)
        seg, contour = self.net(x)
        seg, contour = seg.float(), contour.float()
        # engine.py: alpha = sigmoid(epoch / 100 * 20 - 10)  (consistency-loss ramp-up)
        alpha = torch.sigmoid(torch.tensor(epoch / 100 * 20 - 10.0))
        with torch.autocast("cuda", enabled=False):
            return self.bd_utils.Caculate_multi_task_loss(seg, contour, y[:, None], cont, alpha)


def build(name):
    if name == "unet":
        return SMPWrapper("unet", "resnet34")
    if name == "unet_r50":
        return SMPWrapper("unet", "resnet50")
    if name == "unetpp":
        return SMPWrapper("unetpp", "resnet34")
    if name == "deeplabv3p":
        return SMPWrapper("deeplabv3p", "resnet50")
    if name == "segformer":
        return SMPWrapper("segformer", "mit_b2")
    if name == "segformer_b4":
        return SMPWrapper("segformer", "mit_b4")
    if name == "lbunet":
        return LBUNetWrapper()
    if name == "bdformer":
        return BDFormerWrapper()
    raise ValueError(name)
