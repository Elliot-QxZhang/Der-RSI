"""Prompt / correction policy for SAM and its batched GT-free executor.

A policy is a JSON-serialisable dict:
{
  "views":  ["id", "h", "v", "hv"],              # test-time views whose predictions are averaged
  "rounds": [ROUND, ...],                       # refinement rounds after the automatic round 0
  "rules":  [RULE, ...],                        # state-dependent overrides (evaluated every round)
  "post":   {"thr": 0.5, "fill": true, "lcc": "largest"|"none"|"keep_big", "keep_frac": 0.1}
}
ROUND = {"box": true, "expand": 0.05, "mask": true, "mask_src": "own"|"consensus",
         "pos": 1, "pos_src": "deep"|"peak", "neg": 0, "neg_src": "box_bg"|"ring",
         "accept": "always"|"piou"|"blend", "margin": 0.0, "blend": 0.5}
RULE  = {"round": r or -1 (any) or "post", "if": [[feature, "<"|">", value], ...],  # conjunction
         "set": {param: value, ...}, "note": "free text"}
Features are the GT-free descriptors of dermrsi/rsi/descriptors.py, computed from the image and
the current consensus prediction. Rules are applied in order; later rules override earlier ones.
"""
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F

from dermrsi.prompts import S, distance_inside, mask_box, peak_point, sample_points, scale_box
from dermrsi.rsi.descriptors import pred_descriptors

ROUND_DEFAULT = {"box": True, "box_src": "pred", "expand": 0.05, "mask": True, "mask_src": "own", "pos": 0,
                 "pos_src": "deep",
                 "neg": 0, "neg_src": "box_bg", "accept": "always", "margin": 0.0, "blend": 0.5, "skip": False}
POST_DEFAULT = {"thr": 0.5, "fill": True, "lcc": "none", "keep_frac": 0.1, "coarse_w": 0.0}
# round-0 prompt: learned token / coarse box / both; m_src = same choice for the second expert's views
INIT_DEFAULT = {"src": "auto", "expand": 0.02, "w": 0.5, "m_src": "coarse"}
# per-image ensemble weights of the expert groups (S = SAM2 views, M/Z/P = extra experts); rules with
# round "init" can change them depending on GT-free descriptors (e.g. acquisition resolution)
GROUPS = ["S", "M", "Z", "P"]
GW_DEFAULT = {f"gw_{g}": 1.0 for g in GROUPS}
VIEWS = ["id", "h", "v", "hv"]
# additional SAM-family experts contribute views "<letter>_<view>": e.g. "M_id" (expert M, identity view).
# Expert letters: M = MedSAM (LoRA), Z = MedSAM zero-shot (frozen), P = MedSAM prompt-learning.
EXPERT_PREFIX = "M_"
EXPERT_KEYS = ["M", "Z", "P"]
ALL_VIEWS = VIEWS + [f"{k}_{v}" for k in EXPERT_KEYS for v in VIEWS]


def view_expert(v):
    return v.split("_", 1)[0] if "_" in v else None


def base_view(v):
    return v.split("_", 1)[1] if "_" in v else v

PARAM_SPACE = {  # used for validation of proposals and by the mutation operators
    "box": [False, True], "box_src": ["pred", "coarse", "union"], "expand": (-0.05, 0.25), "mask": [False, True], "mask_src": ["own", "consensus"],
    "pos": [0, 1, 2, 3], "pos_src": ["deep", "peak"], "neg": [0, 1, 2], "neg_src": ["box_bg", "ring"],
    "accept": ["always", "piou", "blend"], "margin": (-0.05, 0.05), "blend": (0.0, 1.0), "skip": [False, True],
    "thr": (0.2, 0.8), "fill": [False, True], "lcc": ["none", "largest", "keep_big"], "keep_frac": (0.0, 0.5),
    "coarse_w": (0.0, 0.6), "src": ["auto", "coarse", "both"], "w": (0.0, 1.0),
    "gw_S": (0.0, 3.0), "gw_M": (0.0, 3.0), "gw_Z": (0.0, 3.0), "gw_P": (0.0, 3.0),
}
CAT = {"box_src": ["pred", "coarse", "union"], "mask_src": ["own", "consensus", "coarse"], "pos_src": ["deep", "peak"], "neg_src": ["box_bg", "ring"],
       "accept": ["always", "piou", "blend"], "lcc": ["none", "largest", "keep_big"]}


def default_policy(n_rounds=2, init="auto"):
    return {"views": ["id"], "init": dict(INIT_DEFAULT, src=init),
            "rounds": [dict(ROUND_DEFAULT) for _ in range(n_rounds)], "rules": [], "post": dict(POST_DEFAULT)}


def validate(policy):
    """Clamp / repair a (possibly LLM-proposed) policy so it is always executable."""
    p = copy.deepcopy(policy)
    p.setdefault("views", ["id"])
    p["views"] = [v for v in dict.fromkeys(p["views"]) if v in ALL_VIEWS] or ["id"]
    if "view_w" in p:  # optional per-view averaging weights
        p["view_w"] = {v: float(min(max(float(p["view_w"].get(v, 1.0)), 0.0), 3.0)) for v in p["views"]}
    p["rounds"] = [dict(ROUND_DEFAULT, **{k: v for k, v in r.items() if k in ROUND_DEFAULT})
                   for r in p.get("rounds", [])][:4]
    p["post"] = dict(POST_DEFAULT, **{k: v for k, v in p.get("post", {}).items() if k in POST_DEFAULT})
    p["init"] = dict(INIT_DEFAULT, **{k: v for k, v in p.get("init", {}).items() if k in INIT_DEFAULT})
    p["init"]["expand"] = float(min(max(float(p["init"]["expand"]), -0.05), 0.25))
    p["init"]["w"] = _clamp("w", p["init"]["w"])
    p["init"]["src"] = _clamp("src", p["init"]["src"])
    p["init"]["m_src"] = _clamp("src", p["init"]["m_src"])
    for d in p["rounds"] + [p["post"]]:
        for k, v in d.items():
            d[k] = _clamp(k, v)
    rules = []
    for r in p.get("rules", []):
        if not isinstance(r, dict) or "set" not in r:
            continue
        conds = [c for c in r.get("if", []) if isinstance(c, (list, tuple)) and len(c) == 3 and c[1] in "<>"]
        st = {k: _clamp(k, v) for k, v in r["set"].items() if k in PARAM_SPACE}
        if st:
            rules.append({"round": r.get("round", -1), "if": [[c[0], c[1], float(c[2])] for c in conds],
                          "set": st, "note": r.get("note", "")})
    p["rules"] = rules[:12]
    return p


def _clamp(k, v):
    sp = PARAM_SPACE.get(k)
    if sp is None:
        return v
    if isinstance(sp, tuple):
        try:
            return float(min(max(float(v), sp[0]), sp[1]))
        except (TypeError, ValueError):
            return float(sp[0])
    if v in sp:
        return v
    if isinstance(sp[0], bool):
        return bool(v)
    if isinstance(sp[0], int):
        try:
            return int(min(max(int(v), min(sp)), max(sp)))
        except (TypeError, ValueError):
            return sp[0]
    return sp[0]


def policy_str(p):
    return json.dumps(p, sort_keys=True)


def load_coarse(split, src="unet_s512"):
    """Coarse localiser probabilities (N,256,256) float16 from runs/baselines/<src>/pred_<split>.npy."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "runs", "baselines", src, f"pred_{split}.npy")
    if not os.path.exists(path):
        return None
    a = torch.from_numpy(np.load(path).astype(np.float32) / 255.0)
    return F.interpolate(a[:, None], size=(256, 256), mode="area")[:, 0].half()


# ---------------------------------------------------------------- view transforms
def _flip_xy(v):
    v = base_view(v)
    return ("h" in v), ("v" in v)


def view_tensor(x, v):
    fx, fy = _flip_xy(v)
    dims = ([-1] if fx else []) + ([-2] if fy else [])
    return x.flip(dims) if dims else x


def view_box(box, v):
    fx, fy = _flip_xy(v)
    b = box.clone()
    if fx:
        b[:, 0], b[:, 2] = S - box[:, 2], S - box[:, 0]
    if fy:
        b[:, 1], b[:, 3] = S - box[:, 3], S - box[:, 1]
    return b


def view_points(pts, v):
    fx, fy = _flip_xy(v)
    p = pts.clone()
    if fx:
        p[..., 0] = S - pts[..., 0]
    if fy:
        p[..., 1] = S - pts[..., 1]
    return p


# ---------------------------------------------------------------- per-image parameter resolution
def _cond_mask(conds, desc, b, dev):
    m = torch.ones(b, dtype=torch.bool, device=dev)
    for f, op, val in conds:
        if f not in desc:
            return torch.zeros(b, dtype=torch.bool, device=dev)
        x = desc[f]
        m &= (x < val) if op == "<" else (x > val)
    return m


def resolve(base, rules, round_key, desc, b, dev):
    """Per-image parameter tensors for one round (or 'post')."""
    out = {}
    for k, v in base.items():
        if k in CAT:
            out[k] = torch.full((b,), CAT[k].index(v), dtype=torch.long, device=dev)
        elif isinstance(v, bool):
            out[k] = torch.full((b,), v, dtype=torch.bool, device=dev)
        elif isinstance(v, int):
            out[k] = torch.full((b,), v, dtype=torch.long, device=dev)
        else:
            out[k] = torch.full((b,), float(v), dtype=torch.float32, device=dev)
    for r in rules:
        if r["round"] != round_key and not (r["round"] == -1 and round_key not in ("post", "init")):
            continue
        m = _cond_mask(r["if"], desc, b, dev)
        if not m.any():
            continue
        for k, v in r["set"].items():
            if k not in out:
                continue
            val = CAT[k].index(v) if k in CAT else v
            out[k] = torch.where(m, torch.as_tensor(val, dtype=out[k].dtype, device=dev), out[k])
    return out


# ---------------------------------------------------------------- executor
class Executor:
    """Runs a policy on cached SAM features. feats[v] = list of 3 tensors (N, C, H, W) for view v."""

    def __init__(self, model, feats, lab, img_desc, device="cuda", coarse=None, expert=None):
        """coarse: optional (N,256,256) float tensor in [0,1] from the lightweight coarse localiser.
        expert: optional dict letter -> SAM-family model; its views are keyed "<letter>_<view>" in feats
                (a bare model is treated as expert "M")."""
        self.model = model
        self.experts = expert if isinstance(expert, dict) else ({"M": expert} if expert is not None else {})
        self.coarse = coarse
        self.feats = feats
        self.lab = lab
        self.img_desc = img_desc  # dict name -> (N,) numpy
        self.dev = device
        self.n = next(iter(feats.values()))[0].shape[0]
        self._r0 = {}

    def _m(self, v):
        e = view_expert(v)
        return self.experts[e] if e else self.model

    def _f(self, v, idx):
        return [f[idx].to(self.dev, non_blocking=True) for f in self.feats[v]]

    def _coarse(self, idx):
        if self.coarse is None:
            return None
        return self.coarse[idx].to(self.dev).float()

    @torch.no_grad()
    def _round0(self, v, idx, init):
        src = init["m_src"] if view_expert(v) else init["src"]
        src = src if self.coarse is not None else "auto"
        key = (v, int(idx[0]), len(idx), src, round(init["expand"], 4), round(init["w"], 4))
        if key not in self._r0:
            f = self._f(v, idx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if src in ("auto", "both"):
                    lg_a, iou_a = self._m(v).decode(f)
                    lg_a, iou_a = view_tensor(lg_a.float(), v), iou_a.float()
                if src in ("coarse", "both"):
                    box = scale_box(mask_box(self._coarse(idx) > 0.5), init["expand"])
                    lg_c, iou_c = self._m(v).decode(f, boxes=view_box(box, v))
                    lg_c, iou_c = view_tensor(lg_c.float(), v), iou_c.float()
            if src == "auto":
                out = (lg_a, iou_a)
            elif src == "coarse":
                out = (lg_c, iou_c)
            else:  # probability-space blend of the two entry prompts
                w = init["w"]
                pa, pc = torch.sigmoid(lg_a), torch.sigmoid(lg_c)
                out = (torch.logit((w * pc + (1 - w) * pa).clamp(1e-4, 1 - 1e-4)), w * iou_c + (1 - w) * iou_a)
            self._r0[key] = (out[0].to(torch.bfloat16).cpu(), out[1].cpu())  # CPU cache: bounded GPU memory
        lg, iou = self._r0[key]
        return lg.to(self.dev).float(), iou.to(self.dev)

    def _desc(self, P, piou, view_lg, idx):
        lab = self.lab[idx].to(self.dev)
        d = pred_descriptors(P, lab, piou, torch.stack([(lg[:, 0] > 0) for lg in view_lg]))
        c = self._coarse(idx)
        if c is not None:
            a, bm = P > 0.5, c > 0.5
            inter = (a & bm).flatten(1).sum(1).float()
            uni = (a | bm).flatten(1).sum(1).float()
            d["coarse_iou"] = torch.where(uni > 0, inter / uni.clamp(min=1), torch.ones_like(uni))
        for k, v in self.img_desc.items():
            d[k] = torch.as_tensor(v[idx.numpy()], device=self.dev, dtype=torch.float32)
        return d

    @torch.no_grad()
    def run_batch(self, policy, idx, trace=False):
        views = policy["views"]
        b = len(idx)
        dev = self.dev
        view_lg, ious = [], []
        init = policy.get("init", INIT_DEFAULT)
        for v in views:
            lg, iou = self._round0(v, idx, init)
            view_lg.append(lg)
            ious.append(iou)
        vw = torch.tensor([policy.get("view_w", {}).get(v, 1.0) for v in views], device=dev)
        W = vw[:, None].expand(len(views), b)  # (V,B) per-image view weights

        def vmean(xs):
            Wn = W / W.sum(0, keepdim=True).clamp(min=1e-6)
            return (torch.stack(xs) * Wn.view(*Wn.shape, *([1] * (xs[0].dim() - 1)))).sum(0)

        P = vmean([torch.sigmoid(l[:, 0]) for l in view_lg])
        piou = vmean(ious)
        if any(r["round"] == "init" for r in policy["rules"]):
            gw = resolve(GW_DEFAULT, policy["rules"], "init", self._desc(P, piou, view_lg, idx), b, dev)
            grp = [view_expert(v) or "S" for v in views]
            W = torch.stack([vw[i] * gw[f"gw_{g}"] for i, g in enumerate(grp)])
            W = torch.where(W.sum(0, keepdim=True) > 0, W, vw[:, None].expand_as(W))  # never all-zero
            P = vmean([torch.sigmoid(l[:, 0]) for l in view_lg])
            piou = vmean(ious)
        hist = [P] if trace else None
        for ri, rd in enumerate(policy["rounds"]):
            desc = self._desc(P, piou, view_lg, idx)
            prm = resolve(rd, policy["rules"], ri, desc, b, dev)
            active = ~prm["skip"]
            if not active.any():
                continue
            cons = P > 0.5
            coarse = self._coarse(idx)
            if coarse is not None:
                cm = coarse > 0.5
                bsrc = prm["box_src"].view(b, 1, 1)
                box_mask = torch.where(bsrc == 0, cons, torch.where(bsrc == 1, cm, cons | cm))
            else:
                box_mask = cons
            box = scale_box(mask_box(box_mask), prm["expand"])
            # positive points: deepest interior points ("deep") or most confident ("peak")
            dist = distance_inside(cons)
            score_pos = torch.where((prm["pos_src"] == 0).view(b, 1, 1), dist, P * cons)
            pos, pv = peak_point(score_pos, k=3)
            pv = pv & (torch.arange(3, device=dev)[None] < prm["pos"][:, None])
            # negative points: background inside the box ("box_bg") or ring just outside the mask
            yy = torch.arange(256, device=dev).view(1, 256, 1).float() * S / 256
            xx = torch.arange(256, device=dev).view(1, 1, 256).float() * S / 256
            inbox = (xx >= box[:, 0, None, None]) & (xx <= box[:, 2, None, None]) & \
                    (yy >= box[:, 1, None, None]) & (yy <= box[:, 3, None, None])
            bg_dist = distance_inside(~cons & inbox)
            ring = (F.max_pool2d(cons.float()[:, None], 25, 1, 12)[:, 0] > 0) & ~cons
            score_neg = torch.where((prm["neg_src"] == 0).view(b, 1, 1), bg_dist, ring.float() * (1 - P))
            neg, nv = peak_point(score_neg, k=2)
            nv = nv & (torch.arange(2, device=dev)[None] < prm["neg"][:, None])
            pts = torch.cat([pos, neg], 1)
            labs = torch.cat([torch.where(pv, 1, -10), torch.where(nv, 0, -10)], 1)
            use_pts = (labs >= 0).any()
            cons_logit = vmean(view_lg)  # consensus over views (already un-flipped)
            new_lg, new_iou = [], []
            if coarse is not None:
                coarse_logit = torch.logit(coarse.clamp(1e-3, 1 - 1e-3))[:, None] * 0.5  # tempered
            else:
                coarse_logit = cons_logit
            for vi, v in enumerate(views):
                ms = prm["mask_src"].view(b, 1, 1, 1)
                mlog = torch.where(ms == 0, view_lg[vi], torch.where(ms == 1, cons_logit, coarse_logit))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, iou = self._m(v).decode(
                        self._f(v, idx), boxes=view_box(box, v), box_on=prm["box"],
                        points=view_points(pts, v) if use_pts else None, labels=labs if use_pts else None,
                        mask_logits=view_tensor(mlog, v), mask_on=prm["mask"],
                        use_auto=~(prm["box"] | (labs >= 0).any(1)))
                new_lg.append(view_tensor(lg.float(), v))
                new_iou.append(iou.float())
            P_new = vmean([torch.sigmoid(l[:, 0]) for l in new_lg])
            piou_new = vmean(new_iou)
            # acceptance
            acc = prm["accept"]
            take = torch.ones(b, dtype=torch.bool, device=dev)
            take = torch.where(acc == 1, piou_new >= piou + prm["margin"], take)
            take &= active
            w = torch.where(acc == 2, prm["blend"], torch.ones_like(prm["blend"]))
            w = torch.where(take, w, torch.zeros_like(w))
            P = w.view(b, 1, 1) * P_new + (1 - w.view(b, 1, 1)) * P
            piou = w * piou_new + (1 - w) * piou
            view_lg = [torch.where(take.view(b, 1, 1, 1), nl, ol) for nl, ol in zip(new_lg, view_lg)]
            if trace:
                hist.append(P)
        desc = self._desc(P, piou, view_lg, idx)
        post = resolve(policy["post"], policy["rules"], "post", desc, b, dev)
        c = self._coarse(idx)
        if c is not None:
            w = post["coarse_w"].view(b, 1, 1)
            P = (1 - w) * P + w * c
        return P, post, desc, hist

    def run(self, policy, bs=32, trace=False):
        """Returns consensus probabilities (N,256,256) on CPU, post params (list of dicts), descriptors."""
        Ps, posts, descs, hists = [], [], [], []
        for s in range(0, self.n, bs):
            idx = torch.arange(s, min(self.n, s + bs))
            P, post, desc, hist = self.run_batch(policy, idx, trace)
            Ps.append(P.cpu())
            posts.append({k: v.cpu() for k, v in post.items()})
            descs.append({k: v.cpu() for k, v in desc.items()})
            if trace:
                hists.append([h.cpu() for h in hist])
        P = torch.cat(Ps)
        post = {k: torch.cat([p[k] for p in posts]) for k in posts[0]}
        desc = {k: torch.cat([d[k] for d in descs]).numpy() for k in descs[0]}
        return P, post, desc, hists
