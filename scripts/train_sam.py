"""Stage A: adapt SAM 2.1 (frozen weights + LoRA + decoder) to the multi-round prompt loop.

Each iteration encodes the image once and decodes R rounds:
  round 0 : learned automatic lesion token only (fully automatic entry point)
  round r : prompts derived from round r-1 -- a box (from the previous prediction, or a noisy GT box),
            the previous logits as dense mask prompt, and correction clicks. Clicks are sampled either
            from true error regions (FN -> positive, FP -> negative; teaches SAM the meaning of
            corrective clicks) or from the model's own uncertain band (the kind of clicks the RSI policy
            generates at test time, where GT is unavailable).
The IoU head is supervised so that the RSI policy can use predicted quality without GT.

torchrun --nproc_per_node 4 scripts/train_sam.py --variant large --epochs 40
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dermrsi.data import SegDataset  # noqa: E402
from dermrsi.prompts import down, mask_box, pack_points, sample_points, scale_box  # noqa: E402
from dermrsi.sam_model import DermSAM  # noqa: E402
from dermrsi.sam_infer import evaluate_dev  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def seg_loss(logits, y512):
    lg = F.interpolate(logits.float(), size=y512.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    bce = F.binary_cross_entropy_with_logits(lg, y512)
    p = torch.sigmoid(lg).flatten(1)
    t = y512.flatten(1)
    dice = 1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)
    return bce + dice.mean(), lg


def iou_target(lg, y512):
    p = lg > 0
    t = y512 > 0.5
    inter = (p & t).flatten(1).sum(1).float()
    union = (p | t).flatten(1).sum(1).float()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))


def next_prompts(prev_lg, y512, gen, p_gt_box, p_err_click):
    """Build round-r prompts from the previous round (batch-level choice of prompt composition)."""
    b = prev_lg.shape[0]
    dev = prev_lg.device
    pred = prev_lg > 0
    r = torch.rand(4, generator=gen, device=dev)
    boxes = None
    if r[0] < p_gt_box:
        boxes = scale_box(mask_box(y512 > 0.5), 0.0, jitter=0.08, gen=gen)
    elif r[0] < 0.85:
        ratio = (torch.rand(b, generator=gen, device=dev) * 0.2 - 0.05)
        boxes = scale_box(mask_box(pred), ratio, jitter=0.03, gen=gen)
    mask_prompt = down(prev_lg) if r[1] < 0.6 else None
    points = labels = None
    if r[2] < 0.75:
        k = int(torch.randint(1, 4, (1,), generator=gen, device=dev))
        if r[3] < p_err_click:
            fn = (~pred) & (y512 > 0.5)
            fp = pred & (y512 <= 0.5)
            pos, pv = sample_points(fn, k, gen)
            neg, nv = sample_points(fp, k, gen)
        else:
            prob = torch.sigmoid(prev_lg)
            band = (prob > 0.2) & (prob < 0.8)
            pos, pv = sample_points(prob > 0.9, k, gen)
            neg, nv = sample_points((prob < 0.1) & (~band), k, gen)
        points, labels = pack_points(pos, pv, neg, nv)
    return boxes, points, labels, mask_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="large")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=4, help="per GPU")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--p_gt_box", type=float, default=0.25)
    ap.add_argument("--p_err_click", type=float, default=0.5)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lowres_p", type=float, default=0.0)
    ap.add_argument("--tune", default="lora", choices=["lora", "prompt"])
    ap.add_argument("--n_ctx", type=int, default=4)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    name = (f"sam2{args.variant}_r{args.lora_r}_e{args.epochs}{args.tag}" if args.tune == "lora"
            else f"{args.variant}_prompt{args.n_ctx}_e{args.epochs}{args.tag}")
    out_dir = os.path.join(ROOT, "runs", "sam", name)
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "log.txt"), "a") if rank == 0 else None

    def P(*a):
        if rank == 0:
            s = " ".join(str(x) for x in a)
            print(s, flush=True)
            log.write(s + "\n")
            log.flush()

    P("args", json.dumps(vars(args)))
    model = DermSAM(args.variant, lora_r=args.lora_r, lora_alpha=args.lora_r, tune=args.tune,
                    n_ctx=args.n_ctx).cuda()
    P("trainable(M)", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)

    ds = SegDataset("train", 1024, out_size=1024, train=True, seed=args.seed, lowres_p=args.lowres_p)
    sampler = DistributedSampler(ds, shuffle=True, seed=args.seed, drop_last=True)
    dl = DataLoader(ds, batch_size=args.bs, sampler=sampler, num_workers=8, drop_last=True, pin_memory=True)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:  # identical initialisation on all ranks
        dist.broadcast(p.data, 0)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    total = args.epochs * len(dl)
    warm = min(300, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1000 + args.seed * 97 + rank)

    best = -1
    for ep in range(args.epochs):
        sampler.set_epoch(ep)
        ds.set_epoch(ep)
        model.train()
        t0 = time.time()
        stats = np.zeros(args.rounds + 1)
        n = 0
        for x, y, _ in dl:
            x = x.cuda(non_blocking=True)
            y512 = F.avg_pool2d(y.cuda(non_blocking=True)[:, None], 2)[:, 0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = model.encode(x)
                loss = 0.0
                prev = None
                for r in range(args.rounds):
                    if r == 0:
                        lg, iou = model.decode(feats)
                    else:
                        boxes, pts, labs, mprompt = next_prompts(prev, y512, gen, args.p_gt_box, args.p_err_click)
                        lg, iou = model.decode(feats, boxes=boxes, points=pts, labels=labs, mask_logits=mprompt)
                    sl, lg_up = seg_loss(lg, y512)
                    il = F.mse_loss(iou.float(), iou_target(lg_up.detach(), y512))
                    loss = loss + sl + il
                    stats[r] += sl.item()
                    prev = lg_up.detach()
                loss = loss / args.rounds
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            flat = torch.cat([g.flatten() for g in grads])
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            off = 0
            for p in params:
                p.grad = flat[off:off + p.numel()].view_as(p)
                off += p.numel()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            stats[-1] += loss.item()
            n += 1
        P(f"ep {ep} loss {stats[-1]/n:.4f} rounds " + " ".join(f"{s/n:.4f}" for s in stats[:-1]) +
          f" lr {sched.get_last_lr()[0]:.2e} t {time.time()-t0:.0f}s")
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            if rank == 0:
                model.eval()
                res = evaluate_dev(model)
                model.train()
                msg = "  dev " + " ".join(f"{k} {v*100:.2f}" for k, v in res.items())
                score = res["r0_dice"]
                torch.save(model.trainable_state_dict(), os.path.join(out_dir, "last.pt"))
                if score > best:
                    best = score
                    torch.save(model.trainable_state_dict(), os.path.join(out_dir, "best.pt"))
                    msg += " *"
                P(msg)
            dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
