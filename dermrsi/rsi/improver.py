"""The self-modifying improver: proposes child policies from a parent policy and a failure report.

The improver is itself part of the inherited genome (Darwin-Godel-Machine style). Its state holds
  op_weights : sampling weights over edit operators
  fix_prefs  : per failure type, a running score for each (param, direction) edit
  focus      : how failure modes are chosen for targeting ("mass" | "lift" | "uniform")
  fail_thr   : 1-Dice threshold that defines a failure in the analysis
  step       : mutation step scale
  memory     : short list of lessons (edit -> observed gain), shown to the LLM proposer
After every generation `update()` credits the operators / edits that produced accepted gains, and
`meta_mutate()` randomly perturbs focus / fail_thr / step / op_weights; children inherit the
updated improver, and parents whose improvers yielded more progress are selected more often.
"""
import copy
import json
import random

from dermrsi.rsi.policy import PARAM_SPACE, ROUND_DEFAULT, validate

OPS = ["targeted_rule", "round_mutation", "post_mutation", "add_round", "drop_rule", "views", "init_mutation",
       "view_weight", "expert_rule", "llm"]

# Candidate fixes per failure type: (param, value-or-delta). Deltas are scaled by improver.step.
FIXES = {
    "under": [("expand", +0.05), ("pos", +1), ("thr", -0.08), ("mask", False), ("mask_src", "consensus"),
              ("accept", "always"), ("neg", -1), ("lcc", "none"), ("box", True), ("fill", True),
              ("box_src", "union"), ("coarse_w", +0.15)],
    "over": [("expand", -0.04), ("neg", +1), ("neg_src", "ring"), ("thr", +0.08), ("lcc", "largest"),
             ("lcc", "keep_big"), ("accept", "piou"), ("pos", -1), ("mask", True), ("skip", True),
             ("box_src", "coarse"), ("mask_src", "coarse"), ("coarse_w", +0.15)],
}
ROUND_PARAMS = [k for k in ROUND_DEFAULT]
POST_PARAMS = ["thr", "fill", "lcc", "keep_frac", "coarse_w"]


def default_improver(use_llm=False):
    w = {op: 1.0 for op in OPS}
    w["add_round"] = 0.3
    w["drop_rule"] = 0.3
    w["views"] = 0.4
    w["init_mutation"] = 0.5
    w["view_weight"] = 0.5
    w["expert_rule"] = 1.0
    w["llm"] = 1.5 if use_llm else 0.0
    return {"op_weights": w, "fix_prefs": {t: {} for t in FIXES}, "focus": "mass", "fail_thr": 0.10,
            "step": 1.0, "memory": [], "generation_gain": [], "n_children": 6}


def _edit_key(p, v):
    return f"{p}:{v}"


def _apply(d, param, val, step):
    """Apply a fix to a parameter dict; numeric deltas are added, categorical values set."""
    cur = d.get(param)
    sp = PARAM_SPACE[param]
    if isinstance(sp, tuple):
        d[param] = cur + val * step if isinstance(val, (int, float)) and not isinstance(val, bool) else val
    elif isinstance(sp[0], bool):
        d[param] = bool(val)
    elif isinstance(sp[0], int):
        d[param] = int(round(cur + val)) if isinstance(cur, int) and not isinstance(val, bool) else val
    else:
        d[param] = val
    return d


class Improver:
    def __init__(self, state, rng):
        self.s = state
        self.rng = rng

    # ------------------------------------------------------------ choosing what to fix
    def pick_mode(self, report):
        if not report:
            return None
        if self.s["focus"] == "uniform":
            return self.rng.choice(report)
        key = (lambda r: r["mass"]) if self.s["focus"] == "mass" else (lambda r: r["lift"] * (r["n_fail"] > 4))
        ws = [max(1e-3, key(r)) for r in report]
        return self.rng.choices(report, weights=ws, k=1)[0]

    def pick_fix(self, mtype):
        prefs = self.s["fix_prefs"].setdefault(mtype, {})
        cands = FIXES[mtype]
        ws = [max(0.05, 1.0 + prefs.get(_edit_key(p, v), 0.0)) for p, v in cands]
        return self.rng.choices(cands, weights=ws, k=1)[0]

    # ------------------------------------------------------------ operators
    def targeted_rule(self, policy, report):
        mode = self.pick_mode([r for r in report if r["cond"]] or report)
        if mode is None:
            return None, None
        n_fix = 1 if self.rng.random() < 0.6 else 2
        fixes = [self.pick_fix(mode["type"]) for _ in range(n_fix)]
        p = copy.deepcopy(policy)
        n_rounds = len(p["rounds"])
        where = self.rng.choice(["post"] + list(range(n_rounds))) if n_rounds else "post"
        st = {}
        for param, val in fixes:
            if param in POST_PARAMS:
                where_p = "post"
                base = dict(p["post"])
            else:
                if where == "post":
                    where_p = self.rng.randrange(n_rounds) if n_rounds else None
                    if where_p is None:
                        continue
                else:
                    where_p = where
                base = dict(p["rounds"][where_p])
            st_key = (where_p, param)
            base = _apply(base, param, val, self.s["step"])
            st[st_key] = base[param]
        if not st:
            return None, None
        by_round = {}
        for (r, param), v in st.items():
            by_round.setdefault(r, {})[param] = v
        for r, sets in by_round.items():
            p["rules"].append({"round": r, "if": mode["cond"], "set": sets,
                               "note": f"fix {mode['desc']}"})
        edit = {"op": "targeted_rule", "mode": mode["type"], "fixes": [_edit_key(a, b) for a, b in fixes],
                "cond": mode["cond"]}
        return p, edit

    def round_mutation(self, policy, report):
        p = copy.deepcopy(policy)
        if not p["rounds"]:
            return self.add_round(policy, report)
        ri = self.rng.randrange(len(p["rounds"]))
        param = self.rng.choice(ROUND_PARAMS)
        p["rounds"][ri][param] = self._mutate_value(param, p["rounds"][ri][param])
        return p, {"op": "round_mutation", "fixes": [f"r{ri}.{param}={p['rounds'][ri][param]}"]}

    def post_mutation(self, policy, report):
        p = copy.deepcopy(policy)
        param = self.rng.choice(POST_PARAMS)
        p["post"][param] = self._mutate_value(param, p["post"][param])
        return p, {"op": "post_mutation", "fixes": [f"post.{param}={p['post'][param]}"]}

    def add_round(self, policy, report):
        p = copy.deepcopy(policy)
        if len(p["rounds"]) >= 4:
            return self.round_mutation(policy, report)
        src = copy.deepcopy(p["rounds"][-1]) if p["rounds"] else dict(ROUND_DEFAULT)
        p["rounds"].append(src)
        return p, {"op": "add_round", "fixes": [f"rounds={len(p['rounds'])}"]}

    def drop_rule(self, policy, report):
        p = copy.deepcopy(policy)
        if p["rules"]:
            i = self.rng.randrange(len(p["rules"]))
            p["rules"].pop(i)
            return p, {"op": "drop_rule", "fixes": [f"drop rule {i}"]}
        if len(p["rounds"]) > 1:
            p["rounds"].pop()
            return p, {"op": "drop_rule", "fixes": ["drop last round"]}
        return None, None

    def views(self, policy, report):
        p = copy.deepcopy(policy)
        opts = [["id"], ["id", "h"], ["id", "v"], ["id", "h", "v"], ["id", "h", "v", "hv"]]
        experts = self.s.get("experts") or (["M"] if self.s.get("expert") else [])
        if experts:
            # toggle one expert's view set in/out of the ensemble (keeps the current SAM2 views)
            e = self.rng.choice(experts)
            own = [v for v in p["views"] if not v.startswith(e + "_")]
            sets = [[], [e + "_id"], [e + "_id", e + "_h"], [e + "_id", e + "_h", e + "_v", e + "_hv"]]
            cur_e = [v for v in p["views"] if v.startswith(e + "_")]
            cands = [own + st for st in sets if sorted(st) != sorted(cur_e) and (own + st)]
            if self.rng.random() < 0.6 and cands:
                p["views"] = self.rng.choice(cands)
                return p, {"op": "views", "fixes": [f"views={p['views']}"]}
            sam2 = [v for v in p["views"] if "_" not in v]
            opts = [o + [v for v in p["views"] if "_" in v] for o in opts if sorted(o) != sorted(sam2)]
            opts = [o for o in opts if o]
        cur = sorted(p["views"])
        cands = [o for o in opts if sorted(o) != cur]
        p["views"] = self.rng.choice(cands)
        return p, {"op": "views", "fixes": [f"views={p['views']}"]}

    def expert_rule(self, policy, report):
        """Condition-dependent expert weighting: for the images of a failure mode, re-weight one expert group."""
        groups = sorted({(v.split("_", 1)[0] if "_" in v else "S") for v in policy["views"]})
        if len(groups) < 2:
            return self.views(policy, report)
        mode = self.pick_mode([r for r in report if r["cond"]] or report)
        if mode is None or not mode["cond"]:
            return None, None
        g = self.rng.choice(groups)
        val = self.rng.choice([0.0, 0.25, 0.5, 2.0, 3.0])
        p = copy.deepcopy(policy)
        p["rules"].append({"round": "init", "if": mode["cond"], "set": {f"gw_{g}": val},
                           "note": f"{mode['desc']}: weight expert {g} x{val}"})
        return p, {"op": "expert_rule", "mode": mode["type"], "fixes": [f"gw_{g}:{val}"], "cond": mode["cond"]}

    def view_weight(self, policy, report):
        p = copy.deepcopy(policy)
        if len(p["views"]) < 2:
            return self.views(policy, report)
        groups = sorted({(v.split("_", 1)[0] if "_" in v else "S") for v in p["views"]})
        g = self.rng.choice(groups)
        f = self.rng.choice([0.5, 0.75, 1.5, 2.0])
        w = dict(p.get("view_w", {}))
        for v in p["views"]:
            if (v.split("_", 1)[0] if "_" in v else "S") == g:
                w[v] = float(w.get(v, 1.0) * f)
        p["view_w"] = w
        return p, {"op": "view_weight", "fixes": [f"weight[{g}]*={f}"]}

    def init_mutation(self, policy, report):
        p = copy.deepcopy(policy)
        has_exp = bool(self.s.get("experts") or self.s.get("expert"))
        k = self.rng.choice(["src", "expand", "w"] + (["m_src"] if has_exp else []))
        if k in ("src", "m_src"):
            p["init"][k] = self.rng.choice([s for s in ["auto", "coarse", "both"] if s != p["init"][k]])
        elif k == "expand":
            p["init"]["expand"] = p["init"]["expand"] + self.rng.uniform(-0.04, 0.04) * self.s["step"]
        else:
            p["init"]["w"] = p["init"]["w"] + self.rng.uniform(-0.25, 0.25) * self.s["step"]
        return p, {"op": "init_mutation", "fixes": [f"init.{k}={p['init'][k]}"]}

    def _mutate_value(self, param, cur):
        sp = PARAM_SPACE[param]
        if isinstance(sp, tuple):
            width = (sp[1] - sp[0]) * 0.15 * self.s["step"]
            return cur + self.rng.uniform(-width, width)
        cands = [v for v in sp if v != cur]
        return self.rng.choice(cands) if cands else cur

    # ------------------------------------------------------------ proposal
    def propose(self, policy, report, llm=None, n=None):
        n = n or self.s["n_children"]
        out = []
        ops = [o for o in OPS if self.s["op_weights"].get(o, 0) > 0 and (o != "llm" or llm is not None)]
        ws = [self.s["op_weights"][o] for o in ops]
        n_llm = 0
        for _ in range(n * 3):
            if len(out) >= n:
                break
            op = self.rng.choices(ops, weights=ws, k=1)[0]
            if op == "llm":
                n_llm += 1
                continue
            child, edit = getattr(self, op)(policy, report)
            if child is not None:
                out.append((validate(child), edit))
        if n_llm and llm is not None:
            for child, edit in llm.propose(policy, report, self.s, n_llm):
                out.append((validate(child), edit))
        return out

    # ------------------------------------------------------------ self-improvement of the improver
    def update(self, results):
        """results: list of (edit, gain) for this generation's children (gain = child - parent fitness)."""
        lr = 0.5
        for edit, gain in results:
            if edit is None:
                continue
            op = edit["op"]
            g = max(-1.0, min(1.0, gain * 200))  # 0.5% fitness -> 1.0
            self.s["op_weights"][op] = float(min(5.0, max(0.05, self.s["op_weights"].get(op, 1.0) * (1 + lr * g))))
            if op in ("targeted_rule", "expert_rule"):
                prefs = self.s["fix_prefs"].setdefault(edit["mode"], {})
                for k in edit["fixes"]:
                    prefs[k] = float(max(-0.95, min(4.0, prefs.get(k, 0.0) + g)))
            if gain > 0:
                self.s["memory"].append({"edit": edit, "gain": round(gain * 100, 3)})
        self.s["memory"] = sorted(self.s["memory"], key=lambda m: -m["gain"])[:12]
        self.s["generation_gain"].append(max([g for _, g in results], default=0.0))

    def meta_mutate(self):
        s = copy.deepcopy(self.s)
        r = self.rng.random()
        if r < 0.25:
            s["focus"] = self.rng.choice([f for f in ["mass", "lift", "uniform"] if f != s["focus"]])
        elif r < 0.5:
            s["fail_thr"] = float(min(0.3, max(0.03, s["fail_thr"] * self.rng.choice([0.6, 1.5]))))
        elif r < 0.75:
            s["step"] = float(min(3.0, max(0.25, s["step"] * self.rng.choice([0.5, 2.0]))))
        else:
            op = self.rng.choice([o for o in OPS if s["op_weights"][o] > 0])
            s["op_weights"][op] = float(min(5.0, s["op_weights"][op] * self.rng.choice([0.5, 2.0])))
        return s


def improver_str(s):
    return json.dumps({k: v for k, v in s.items() if k != "memory"}, sort_keys=True)
