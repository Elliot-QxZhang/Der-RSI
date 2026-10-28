"""Stage B: offline recursive self-improvement of the SAM prompting policy (SAM frozen).

Archive-based open-ended search (Darwin-Godel-Machine style) over (policy, improver) pairs:
  1. pick a parent from the archive (prob ~ sigmoid(rank-normalised fitness) / (1 + #children))
  2. run the parent policy with SAM on dev-A, analyse dermoscopic failures (GT used offline only)
  3. the parent's *improver* proposes children (rule edits targeting failure modes, mutations, LLM)
  4. children are executed with SAM and scored on dev-A; a child is inherited into the archive only if it
     beats its parent on dev-A AND does not degrade the held-out dev-B
  5. the improver is credited with the gains it produced (operator / fix preferences + lesson memory),
     optionally meta-mutated, and stored with the child -> the way of improving is itself inherited.
The final frozen policy = archive member with the best fitness on the full dev split.

python scripts/rsi_search.py --ckpt runs/sam/<run>/best.pt --variant large --gens 60 --gpu 0
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dermrsi.data import load_split  # noqa: E402
from dermrsi.rsi import failure  # noqa: E402
from dermrsi.rsi.descriptors import image_descriptors, lab_images  # noqa: E402
from dermrsi.rsi.fitness import fitness, masks_from, per_image_scores, summary  # noqa: E402
from dermrsi.rsi.improver import Improver, default_improver  # noqa: E402
from dermrsi.rsi.policy import Executor, default_policy, load_coarse, policy_str, validate  # noqa: E402
from dermrsi.sam_model import DermSAM  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_executor(model, split, views=("id", "h", "v", "hv"), device="cuda", coarse="unet_s512", expert=None,
                   feat_device=None):
    """expert: dict letter -> model (views '<letter>_<view>'). feat_device: where cached features live
    (default: GPU if a single model, pinned CPU memory when experts are used)."""
    fd = feat_device or ("cpu" if expert else device)

    def put(t):
        return t.pin_memory() if fd == "cpu" else t.to(fd)
    feats = {v: [put(t) for t in encode_split_view(model, split, v)] for v in views}
    for k, m in (expert or {}).items():
        for v in views:
            feats[f"{k}_{v}"] = [put(t) for t in encode_split_view(m, split, v)]
    img, _, meta = load_split(split, 512)
    lab = lab_images(img)
    desc = image_descriptors(img, meta)
    return Executor(model, feats, lab, desc, device, coarse=load_coarse(split, coarse) if coarse else None,
                    expert=expert)


def load_model(variant, ckpt):
    """ckpt == 'frozen' -> original pretrained weights only (no LoRA / decoder adaptation)."""
    return DermSAM.from_checkpoint(variant, ckpt)


@torch.no_grad()
def encode_split_view(model, split, v, bs=4, subset=None):
    from torch.utils.data import DataLoader, Subset
    from dermrsi.data import SegDataset
    from dermrsi.rsi.policy import view_tensor
    ds = SegDataset(split, 1024, out_size=1024, train=False)
    if subset is not None:
        ds = Subset(ds, list(subset))
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8)
    out = None
    for x, _, _ in dl:
        x = view_tensor(x.cuda(non_blocking=True), v)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.encode(x)
        if out is None:
            out = [[] for _ in f]
        for i in range(len(f)):
            out[i].append(f[i].to(torch.bfloat16).cpu())
    return [torch.cat(o) for o in out]


def evaluate(ex, policy, gts, with_bf1=True):
    P, post, desc, _ = ex.run(policy)
    masks = masks_from(P, post, size=gts.shape[-1])
    return per_image_scores(masks, gts, with_bf1=with_bf1), desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="large")
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--children", type=int, default=6)
    ap.add_argument("--llm", type=int, default=1)
    ap.add_argument("--llm_device", default="cuda:0")
    ap.add_argument("--tol_b", type=float, default=0.0005, help="allowed dev-B degradation for inheritance")
    ap.add_argument("--meta_p", type=float, default=0.3)
    ap.add_argument("--mode", default="rsi", choices=["rsi", "no_meta", "random"],
                    help="ablation: rsi = full (improver inherited + self-modified); no_meta = fixed improver "
                         "(no credit assignment / meta-mutation, no failure-targeted ops); random = random "
                         "mutation hill-climbing without failure analysis or archive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=0, help="seed of the dev-A / dev-B partition")
    ap.add_argument("--out", default=None)
    ap.add_argument("--init_rounds", type=int, default=0)
    ap.add_argument("--init_policy", default="", help="optional JSON policy to seed the archive root")
    ap.add_argument("--coarse", default="unet_s512", help="coarse localiser run in runs/baselines ('' = none)")
    ap.add_argument("--experts", nargs="*", default=[],
                    help="extra SAM-family experts as LETTER=variant:ckpt (ckpt 'frozen' = zero-shot), "
                         "e.g. M=medsam:runs/sam/x/best.pt Z=medsam:frozen")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    out_dir = args.out or os.path.join(os.path.dirname(args.ckpt), f"rsi_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "log.txt"), "a")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    P("args", json.dumps(vars(args)))
    model = load_model(args.variant, args.ckpt)
    expert = {}
    for spec in args.experts:
        k, rest = spec.split("=", 1)
        var, ck = rest.split(":", 1)
        expert[k] = load_model(var, ck)
    t0 = time.time()
    ex = build_executor(model, "dev", coarse=args.coarse, expert=expert)
    _, gts, meta = load_split("dev", 512)
    gts = np.asarray(gts)
    n = len(meta)
    perm = np.random.RandomState(7 + args.split_seed).permutation(n)
    idx_a, idx_b = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    P(f"features ready in {time.time()-t0:.0f}s; dev-A {len(idx_a)} dev-B {len(idx_b)}")

    llm = None
    if args.llm:
        from dermrsi.rsi.llm import LLMProposer
        llm = LLMProposer(device=args.llm_device)
        P("LLM proposer loaded")

    cache = {}

    def score(policy):
        key = policy_str(policy)
        if key not in cache:
            sc, desc = evaluate(ex, policy, gts)
            cache[key] = (sc, desc, fitness(sc, idx_a), fitness(sc, idx_b), fitness(sc))
        return cache[key]

    root = validate(json.load(open(args.init_policy)) if args.init_policy else default_policy(args.init_rounds))
    sc, desc, fa, fb, fall = score(root)
    imp0 = default_improver(bool(args.llm))
    imp0["experts"] = sorted(expert)
    archive = [{"id": 0, "parent": None, "gen": 0, "policy": root, "improver": imp0,
                "fa": fa, "fb": fb, "fall": fall, "children": 0, "edit": None}]
    P(f"root  A {fa*100:.3f} B {fb*100:.3f} all {fall*100:.3f}", summary(sc))
    history = []
    for gen in range(1, args.gens + 1):
        tg = time.time()
        # ---- parent selection (DGM): performance-weighted, novelty bonus for few children
        fas = np.array([a["fa"] for a in archive])
        z = (fas - np.median(fas)) / (fas.std() + 1e-6)
        w = 1 / (1 + np.exp(-z)) / (1 + np.array([a["children"] for a in archive]))
        if args.mode == "random":  # greedy hill climbing from the current best (no open-ended archive)
            parent = max(archive, key=lambda a: a["fa"])
        else:
            parent = archive[rng.choices(range(len(archive)), weights=w.tolist(), k=1)[0]]
        parent["children"] += 1
        psc, pdesc, pfa, pfb, _ = score(parent["policy"])
        # ---- failure analysis on dev-A (GT used offline only)
        imp = Improver(copy.deepcopy(parent["improver"]), rng)
        if args.mode == "random":
            for op in imp.s["op_weights"]:
                imp.s["op_weights"][op] = 1.0 if op in ("round_mutation", "post_mutation", "views",
                                                        "init_mutation", "add_round") else 0.0
        sub = {k: v[idx_a] for k, v in psc.items()}
        dsub = {k: np.asarray(v)[idx_a] for k, v in pdesc.items()}
        report = failure.analyze(sub, dsub, fail_thr=imp.s["fail_thr"])
        if gen == 1 or gen % 10 == 0:
            P(failure.report_text(report, sub, summary(sub)))
        # ---- proposals
        kids = imp.propose(parent["policy"], report, llm=llm, n=args.children)
        results, accepted = [], []
        for child, edit in kids:
            csc, cdesc, cfa, cfb, cfall = score(child)
            gain = cfa - pfa
            results.append((edit, gain))
            ok = gain > 1e-5 and cfb >= pfb - args.tol_b and policy_str(child) not in \
                {policy_str(a["policy"]) for a in archive}
            if ok:
                accepted.append((child, edit, cfa, cfb, cfall))
        if args.mode == "rsi":
            imp.update(results)
            parent["improver"] = copy.deepcopy(imp.s)  # parent's improver learns from its children
        for child, edit, cfa, cfb, cfall in accepted:
            st = imp.meta_mutate() if (args.mode == "rsi" and rng.random() < args.meta_p) else copy.deepcopy(imp.s)
            archive.append({"id": len(archive), "parent": parent["id"], "gen": gen, "policy": child,
                            "improver": st, "fa": cfa, "fb": cfb, "fall": cfall, "children": 0, "edit": edit})
        best = max(archive, key=lambda a: a["fall"])
        history.append({"gen": gen, "parent": parent["id"], "n_kids": len(kids), "n_acc": len(accepted),
                        "best_fall": best["fall"], "best_id": best["id"],
                        "gains": [round(g * 100, 3) for _, g in results],
                        "ops": [e["op"] if e else None for e, _ in results]})
        P(f"gen {gen} parent {parent['id']} (A {pfa*100:.3f}) kids {len(kids)} acc {len(accepted)} "
          f"gains {[round(g*100,3) for _, g in results]} ops {[e['op'] if e else None for e, _ in results]} | "
          f"archive {len(archive)} best#{best['id']} all {best['fall']*100:.3f} | {time.time()-tg:.0f}s")
        if gen % 5 == 0 or gen == args.gens:
            with open(os.path.join(out_dir, "archive.json"), "w") as f:
                json.dump({"archive": archive, "history": history, "idx_a": idx_a.tolist(),
                           "idx_b": idx_b.tolist()}, f, indent=1)
            if llm is not None:
                with open(os.path.join(out_dir, "llm_log.json"), "w") as f:
                    json.dump(llm.log, f, indent=1)
    best = max(archive, key=lambda a: a["fall"])
    with open(os.path.join(out_dir, "best_policy.json"), "w") as f:
        json.dump(best["policy"], f, indent=1)
    sc = score(best["policy"])[0]
    P("BEST", best["id"], "gen", best["gen"], summary(sc), json.dumps(best["policy"]))
    # lineage of the best policy
    lin, a = [], best
    while a is not None:
        lin.append({"id": a["id"], "gen": a["gen"], "fall": a["fall"], "edit": a["edit"]})
        a = archive[a["parent"]] if a["parent"] is not None else None
    with open(os.path.join(out_dir, "best_lineage.json"), "w") as f:
        json.dump(lin[::-1], f, indent=1)


if __name__ == "__main__":
    main()
