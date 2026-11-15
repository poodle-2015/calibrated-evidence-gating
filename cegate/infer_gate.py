"""Likelihood-based Gate inference.

We never parse generated Gate text. We feed the target up to the decision point and
compare the likelihood of the two labels, "sufficient" and "insufficient". The reported
insufficiency score is

    score = logP("insufficient" | prompt, prefix) - logP("sufficient" | prompt, prefix)

so higher means more insufficient. Calibration then picks a single threshold on this
score (see cegate/calibration.py).

Requires torch, transformers, and peft.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .prompts import render_gate

# Common JSON prefix; we score the AVERAGE log-likelihood of the two label words that
# follow it (paper Section 3.3), not the trailing JSON punctuation.
_PREFIX = '{"sufficiency": "'
_LABELS = {"sufficient": "sufficient", "insufficient": "insufficient"}


def _lazy_imports():
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        from peft import PeftModel  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "infer_gate requires torch, transformers, and peft. "
            "Install the dependencies in requirements.txt."
        ) from e
    return torch, AutoModelForCausalLM, AutoTokenizer, PeftModel


class GateScorer:
    def __init__(self, base_model: str, adapter: Optional[str] = None,
                 dtype: str = "bfloat16", device: str = "cuda"):
        torch, AutoModelForCausalLM, AutoTokenizer, PeftModel = _lazy_imports()
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=getattr(torch, dtype), device_map=device)
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def _build_ids(self, context: str, question: str, continuation: str):
        """Tokenize prompt + assistant prefix + continuation; return ids and the index
        range covering `continuation` for per-token log-prob summation."""
        torch = self.torch
        parts = render_gate(context, question)
        messages = [{"role": "system", "content": parts["system"]},
                    {"role": "user", "content": parts["user"]}]
        prompt_ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt")[0]
        prefix_ids = self.tok(_PREFIX, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        cont_ids = self.tok(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        ids = torch.cat([prompt_ids, prefix_ids, cont_ids])
        start = len(prompt_ids) + len(prefix_ids)
        return ids, start, len(cont_ids)

    def _avg_loglik(self, context: str, question: str, continuation: str) -> float:
        """Average per-token log-likelihood of `continuation` after the shared prefix."""
        torch = self.torch
        ids, start, n = self._build_ids(context, question, continuation)
        with torch.no_grad():
            logits = self.model(ids.unsqueeze(0).to(self.model.device)).logits[0]
        logprobs = torch.log_softmax(logits, dim=-1)
        total = 0.0
        for pos in range(start, start + n):
            total += logprobs[pos - 1, ids[pos]].item()
        return total / max(n, 1)

    def score(self, context: str, question: str) -> float:
        # score = avg logP("insufficient") - avg logP("sufficient"); higher => insufficient
        ll_ins = self._avg_loglik(context, question, _LABELS["insufficient"])
        ll_suf = self._avg_loglik(context, question, _LABELS["sufficient"])
        return ll_ins - ll_suf

    def score_pool(self, examples: Iterable[Dict]) -> List[Dict]:
        out = []
        for example in examples:
            out.append({"id": example["id"],
                        "score": self.score(example["context"], example["question"])})
        return out
