"""Dermoscopic failure analysis: turn per-image errors into a structured, GT-free-actionable report.

Offline we know the GT, so every dev image gets an error type (under- / over-segmentation) and a loss
(1 - Dice). Each failure mode is then *described with GT-free descriptors only*, by searching the
single-feature threshold that best isolates the high-loss images of that mode. The resulting
conditions ("contrast < 9.7", "hair > 0.04", ...) are exactly the conditions that policy rules can
test at inference time, which is what makes the diagnosis actionable without labels.
"""
import numpy as np

from dermrsi.rsi.descriptors import FEATURES

MODE_TYPES = ["under", "over", "any"]


def assign_modes(scores, fail_thr=0.10):
    loss = 1 - scores["dice"]
    fail = loss > fail_thr
    under = fail & (scores["fn_rate"] >= scores["fp_rate"])
    over = fail & (scores["fn_rate"] < scores["fp_rate"])
    return {"under": under, "over": over, "any": fail}, loss


def best_condition(x, target, weight, min_support=6, qs=np.linspace(0.05, 0.95, 19)):
    """Threshold on one feature maximising the loss mass captured per covered image (lift)."""
    best = None
    base = weight[target].sum() / max(1, len(x))
    for q in qs:
        t = float(np.quantile(x, q))
        for op in ["<", ">"]:
            sel = x < t if op == "<" else x > t
            n = int(sel.sum())
            if n < min_support or n > 0.7 * len(x):
                continue
            mass = float(weight[sel & target].sum())
            lift = (mass / n) / max(base, 1e-9)
            score = mass * np.log1p(lift)  # prefer conditions that capture much loss with high lift
            if best is None or score > best["score"]:
                best = {"op": op, "thr": t, "n": n, "mass": mass, "lift": lift, "score": score,
                        "n_fail": int((sel & target).sum())}
    return best


def analyze(scores, desc, top_k=6, fail_thr=0.10):
    """Returns a list of failure modes sorted by explained loss mass."""
    modes, loss = assign_modes(scores, fail_thr)
    report = []
    for mt in ["under", "over"]:
        tgt = modes[mt]
        if tgt.sum() < 3:
            continue
        report.append({"type": mt, "cond": [], "n_fail": int(tgt.sum()), "mass": float(loss[tgt].sum()),
                       "lift": 1.0, "desc": f"all {mt}-segmentation failures"})
        for f in FEATURES:
            if f not in desc:
                continue
            c = best_condition(np.asarray(desc[f], np.float64), tgt, loss)
            if c is None:
                continue
            report.append({"type": mt, "cond": [[f, c["op"], round(c["thr"], 4)]], "n_fail": c["n_fail"],
                           "n_cover": c["n"], "mass": c["mass"], "lift": c["lift"],
                           "desc": f"{mt}-seg where {f} {c['op']} {c['thr']:.3g}"})
    report.sort(key=lambda r: -r["mass"] * np.log1p(r["lift"]))
    # de-duplicate: keep best condition per (type, feature)
    seen, out = set(), []
    for r in report:
        key = (r["type"], r["cond"][0][0] if r["cond"] else None)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out[:top_k]


def report_text(report, scores, summary):
    lines = [f"dev-search: Dice {summary['dice']*100:.2f} IoU {summary['iou']*100:.2f} "
             f"BF1 {summary['bf1']*100:.2f}; failures(Dice<0.9): {(scores['dice'] < 0.9).sum()} / {len(scores['dice'])}; "
             f"total loss mass {float((1 - scores['dice']).sum()):.2f}"]
    for i, r in enumerate(report):
        lines.append(f"  mode {i}: {r['desc']} | failing imgs {r['n_fail']} | loss mass {r['mass']:.2f} | "
                     f"lift {r['lift']:.2f}" + (f" | covers {r.get('n_cover')} imgs" if r["cond"] else ""))
    return "\n".join(lines)
