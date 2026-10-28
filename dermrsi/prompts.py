"""Batched GPU prompt construction from masks / probability maps (all coords in 1024 space)."""
import torch
import torch.nn.functional as F

S = 1024


def mask_box(mask, min_size=8):
    """mask (B,H,W) bool -> (B,4) xyxy in 1024 space; empty masks -> full image box."""
    b, h, w = mask.shape
    ys = mask.any(dim=2)
    xs = mask.any(dim=1)
    ar_h = torch.arange(h, device=mask.device)
    ar_w = torch.arange(w, device=mask.device)
    big = 10 ** 6
    y0 = torch.where(ys, ar_h, big).min(1).values
    y1 = torch.where(ys, ar_h, -1).max(1).values
    x0 = torch.where(xs, ar_w, big).min(1).values
    x1 = torch.where(xs, ar_w, -1).max(1).values
    empty = ~mask.flatten(1).any(1)
    sx, sy = S / w, S / h
    box = torch.stack([x0 * sx, y0 * sy, (x1 + 1) * sx, (y1 + 1) * sy], 1).float()
    full = torch.tensor([0, 0, S, S], device=mask.device, dtype=torch.float32).expand(b, 4)
    box = torch.where(empty[:, None], full, box)
    # enforce a minimum box size
    cx, cy = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
    bw = (box[:, 2] - box[:, 0]).clamp(min=min_size)
    bh = (box[:, 3] - box[:, 1]).clamp(min=min_size)
    return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def scale_box(box, ratio, jitter=0.0, gen=None):
    """Expand (ratio>0) or shrink a box by ratio*size on each side, plus optional random jitter."""
    bw = (box[:, 2] - box[:, 0])[:, None]
    bh = (box[:, 3] - box[:, 1])[:, None]
    if not torch.is_tensor(ratio):
        ratio = torch.full((box.shape[0], 1), float(ratio), device=box.device)
    elif ratio.dim() == 1:
        ratio = ratio[:, None]
    d = torch.cat([-bw, -bh, bw, bh], 1) * ratio
    if jitter > 0:
        n = torch.randn(box.shape, device=box.device, generator=gen) * jitter
        d = d + n * torch.cat([bw, bh, bw, bh], 1)
    return (box + d).clamp(0, S)


def sample_points(region, k, gen=None):
    """Sample k points uniformly from region (B,H,W) bool. Returns xy (B,k,2) in 1024 space and
    a validity mask (B,k) (False where the region is empty)."""
    b, h, w = region.shape
    flat = region.flatten(1).float()
    valid = flat.sum(1) > 0
    probs = torch.where(valid[:, None], flat, torch.ones_like(flat))
    idx = torch.multinomial(probs, k, replacement=True, generator=gen)
    y = (idx // w).float() + 0.5
    x = (idx % w).float() + 0.5
    pts = torch.stack([x * S / w, y * S / h], -1)
    return pts, valid[:, None].expand(b, k)


def peak_point(score, k=1, suppress=0.08):
    """Greedy arg-max points with suppression (deterministic). score (B,H,W). Returns (B,k,2), valid."""
    b, h, w = score.shape
    s = score.clone()
    pts, val = [], []
    yy = torch.arange(h, device=s.device).view(1, h, 1).float()
    xx = torch.arange(w, device=s.device).view(1, 1, w).float()
    r = suppress * max(h, w)
    for _ in range(k):
        m, idx = s.flatten(1).max(1)
        y = (idx // w).float()
        x = (idx % w).float()
        pts.append(torch.stack([(x + 0.5) * S / w, (y + 0.5) * S / h], -1))
        val.append(m > 0)
        d2 = (yy - y.view(b, 1, 1)) ** 2 + (xx - x.view(b, 1, 1)) ** 2
        s = torch.where(d2 < r * r, torch.zeros_like(s), s)
    return torch.stack(pts, 1), torch.stack(val, 1)


def distance_inside(mask, iters=None):
    """Approximate distance-to-boundary inside a mask using repeated erosion (GPU, (B,H,W) bool)."""
    x = mask.float()[:, None]
    acc = torch.zeros_like(x)
    n = iters or max(mask.shape[-1] // 8, 4)
    for _ in range(n):
        acc = acc + x
        x = -F.max_pool2d(-x, 3, 1, 1)
        if x.sum() == 0:
            break
    return acc[:, 0]


def pack_points(pos, pos_valid, neg, neg_valid):
    """Concatenate positive and negative points; invalid points get label -10 (padding)."""
    pts = torch.cat([pos, neg], 1)
    lab = torch.cat([torch.where(pos_valid, 1, -10), torch.where(neg_valid, 0, -10)], 1)
    return pts, lab


def down(x, size=256):
    """(B,H,W) or (B,1,H,W) float -> (B,1,size,size)."""
    if x.dim() == 3:
        x = x[:, None]
    if x.shape[-1] == size:
        return x
    return F.interpolate(x.float(), size=(size, size), mode="bilinear", align_corners=False)
