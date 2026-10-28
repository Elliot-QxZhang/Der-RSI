"""Datasets backed by the resized npy caches produced by scripts/prepare_data.py."""
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def load_split(name, size):
    img = np.load(os.path.join(CACHE, f"{name}_img{size}.npy"), mmap_mode="r")
    gt = np.load(os.path.join(CACHE, f"{name}_gt{size}.npy"), mmap_mode="r")
    with open(os.path.join(CACHE, f"{name}.json")) as f:
        meta = json.load(f)
    return img, gt, meta


def random_affine(img, mask, rng, out_size, scale=(0.75, 1.25), max_rot=180, max_shift=0.1):
    s = img.shape[0]
    ang = rng.uniform(-max_rot, max_rot)
    sc = rng.uniform(*scale) * out_size / s
    m = cv2.getRotationMatrix2D((s / 2, s / 2), ang, sc)
    m[:, 2] += (out_size - s) / 2 + rng.uniform(-max_shift, max_shift, 2) * out_size
    img = cv2.warpAffine(img, m, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, m, (out_size, out_size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return img, mask


def color_jitter(img, rng, b=0.2, c=0.2, s=0.2, h=0.03):
    x = img.astype(np.float32)
    x = x * rng.uniform(1 - b, 1 + b)
    mean = x.mean()
    x = (x - mean) * rng.uniform(1 - c, 1 + c) + mean
    x = np.clip(x, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-h, h) * 180) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1 - s, 1 + s), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


class SegDataset(Dataset):
    """Returns normalized CHW float image, HW float mask, and index."""

    def __init__(self, split, size, out_size=None, train=False, seed=0, lowres_p=0.0):
        self.img, self.gt, self.meta = load_split(split, size)
        self.out_size = out_size or size
        self.train = train
        self.seed = seed
        self.epoch = 0
        self.lowres_p = lowres_p  # simulate low-resolution acquisitions (e.g. 640x480 images upsampled)

    def __len__(self):
        return len(self.meta)

    def set_epoch(self, e):
        self.epoch = e

    def __getitem__(self, i):
        img = np.array(self.img[i])
        mask = np.array(self.gt[i])
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            if rng.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if rng.random() < 0.5:
                img, mask = img[::-1], mask[::-1]
            img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
            if rng.random() < 0.8:
                img, mask = random_affine(img, mask, rng, self.out_size)
            if rng.random() < 0.8:
                img = color_jitter(img, rng)
            if rng.random() < self.lowres_p:
                s = img.shape[0]
                r = int(s * rng.uniform(0.35, 0.7))
                img = cv2.resize(cv2.resize(img, (r, r), interpolation=cv2.INTER_AREA), (s, s),
                                 interpolation=cv2.INTER_LINEAR)
                if rng.random() < 0.5:
                    q = int(rng.integers(55, 95))
                    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
                    img = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
        if img.shape[0] != self.out_size:
            img = cv2.resize(img, (self.out_size, self.out_size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.out_size, self.out_size), interpolation=cv2.INTER_NEAREST)
        x = (img.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(mask.astype(np.float32)), i
