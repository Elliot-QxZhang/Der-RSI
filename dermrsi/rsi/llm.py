"""Open-source LLM policy proposer (Qwen3-VL-8B-Instruct, text-only use).

The LLM receives the parent policy (JSON), the structured dermoscopic failure report, the GT-free
descriptor vocabulary and the improver's memory of edits that helped before, and returns child
policies as JSON. Its outputs are validated/clamped by policy.validate and scored on the dev split
like every other proposal, so it is only a proposal generator, never a judge.
"""
import json
import re

import torch

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

SYSTEM = """You improve an automatic prompting policy for SAM 2 (Segment Anything) on dermoscopic skin-lesion images.
SAM weights are frozen. The policy decides which prompts (box / points / previous mask) are given to SAM in each
refinement round and how the final mask is post-processed. You see a failure report computed on a development set.
Return ONLY a JSON list of 1-3 complete child policies (no prose). Each child must keep the schema of the parent.

Schema:
- init: {src: "auto"|"coarse"|"both" (round-0 prompt: learned lesion token, box from a lightweight coarse
  localiser, or a blend), expand: [-0.05,0.25] (coarse box margin), w: [0,1] (blend weight of the coarse-box pass),
  m_src: same choice for the second expert's "M_*" views}
- views: subset of ["id","h","v","hv"] (test-time flips of SAM2 that are averaged); extra SAM-family experts add
  views "<E>_id","<E>_h","<E>_v","<E>_hv" with E in {M: MedSAM-LoRA, Z: MedSAM zero-shot, P: MedSAM prompt-learning}
  (only the experts listed in the parent's views or in the report are available)
- view_w (optional): {view: weight in [0,3]} weights for averaging the views
- rules with round "init" may set per-image expert-group weights gw_S (SAM2 views), gw_M, gw_Z, gw_P in [0,3],
  e.g. trust a different expert for low-resolution acquisitions (feature res) or hairy images
- rounds: list (max 4) of {box: bool, box_src: "pred"|"coarse"|"union", expand: float in [-0.05,0.25] (box margin relative to size),
  mask: bool (feed previous logits as dense prompt), mask_src: "own"|"consensus"|"coarse", pos: 0-3 positive clicks,
  pos_src: "deep"|"peak", neg: 0-2 negative clicks, neg_src: "box_bg"|"ring", accept: "always"|"piou"|"blend",
  margin: float in [-0.05,0.05], blend: float in [0,1], skip: bool}
- rules: list (max 12) of {round: int round index or -1 (all rounds) or "post", if: [[feature, "<" or ">", value], ...],
  set: {param: value}, note: str}. Features (GT-free, available at test time): hair, vignette, lum, res
  (log10 megapixels of the original image), area, border, unc, contrast, compact, piou, tta_dis, coarse_iou
  (agreement between SAM and the coarse localiser).
- post: {thr: [0.2,0.8], fill: bool, lcc: "none"|"largest"|"keep_big", keep_frac: [0,0.5],
  coarse_w: [0,0.6] (weight of the coarse localiser probability in the final fusion)}
Dermoscopy hints: low-contrast / faded pigment edges tend to be under-segmented; hair, rulers and dark
vignetting corners cause over-segmentation; very large lesions touch the image border; fragmented masks have
high compactness ratio."""


class LLMProposer:
    def __init__(self, device="cuda:0", max_new_tokens=1800, temperature=0.8):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16).to(device).eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.log = []

    def _chat(self, prompt):
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=True,
                                      temperature=self.temperature, top_p=0.95)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    @staticmethod
    def _parse(text):
        m = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
        if not m:
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                return []
            cand = "[" + m.group(0) + "]"
        else:
            cand = m.group(0)
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            return []
        return [o for o in obj if isinstance(o, dict)] if isinstance(obj, list) else []

    def propose(self, policy, report, improver_state, n):
        rep = "\n".join(f"- {r['desc']} (failing {r['n_fail']} imgs, loss mass {r['mass']:.2f}, lift {r['lift']:.2f})"
                        for r in report)
        mem = "\n".join(f"- {json.dumps(m['edit'])} -> +{m['gain']:.3f} fitness pts"
                        for m in improver_state.get("memory", [])[:8]) or "- (none yet)"
        ex = improver_state.get("experts") or []
        avail = f"Available extra experts: {ex}\n\n" if ex else "No extra experts are available (use SAM2 views only).\n\n"
        prompt = (f"{avail}Parent policy:\n{json.dumps(policy)}\n\nFailure report on dev set (conditions use GT-free "
                  f"features):\n{rep}\n\nEdits that improved fitness before:\n{mem}\n\nPropose {min(3, n)} "
                  f"different children that target the largest failure modes. JSON list only.")
        try:
            text = self._chat(prompt)
        except Exception as e:  # never let the proposer crash the search
            self.log.append({"error": str(e)})
            return []
        kids = self._parse(text)[:n]
        self.log.append({"prompt_chars": len(prompt), "n_children": len(kids), "raw": text[:4000]})
        return [(k, {"op": "llm", "fixes": [k.get("rules", [{}])[-1].get("note", "llm")] if k.get("rules") else ["llm"]})
                for k in kids]
