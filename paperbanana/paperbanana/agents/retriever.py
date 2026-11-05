"""Retriever: find structurally similar reference figures (lexical recall + optional LLM rerank)."""

from __future__ import annotations

from ..data.bench import Reference, ReferencePool
from ..prompts import RETRIEVER_SYSTEM
from ..providers.base import Message, TextModel
from ..utils.text import extract_json, truncate


class Retriever:
    def __init__(self, pool: ReferencePool | None, text_model: TextModel | None = None, *, llm_rerank: bool = True):
        self.pool = pool
        self.text_model = text_model
        self.llm_rerank = llm_rerank

    def retrieve(self, method_text: str, intent: str, top_k: int = 3) -> list[Reference]:
        if not self.pool or len(self.pool) == 0 or top_k <= 0:
            return []
        query = f"{intent}\n{intent}\n{method_text}"
        candidates = self.pool.search(query, top_k=max(top_k * 4, 8))
        if not candidates:
            return []
        if not (self.llm_rerank and self.text_model) or len(candidates) <= top_k:
            return candidates[:top_k]
        try:
            return self._rerank(method_text, intent, candidates, top_k)
        except Exception:  # noqa: BLE001 - reranking is best-effort
            return candidates[:top_k]

    def _rerank(self, method_text: str, intent: str, candidates: list[Reference], top_k: int) -> list[Reference]:
        listing = "\n".join(
            f"[{i + 1}] ({c.category or 'n/a'}) {c.caption_short(300)}" for i, c in enumerate(candidates)
        )
        prompt = (
            f"TARGET CAPTION:\n{intent}\n\nTARGET METHOD TEXT (excerpt):\n{truncate(method_text, 3500)}\n\n"
            f"CANDIDATE REFERENCES:\n{listing}\n\nSelect the best {top_k} candidates."
        )
        raw = self.text_model.complete([Message.user(prompt)], system=RETRIEVER_SYSTEM, json_mode=True, temperature=0.2)
        data = extract_json(raw)
        picked: list[Reference] = []
        for n in data.get("selected", []):
            try:
                idx = int(n) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(candidates) and candidates[idx] not in picked:
                picked.append(candidates[idx])
        for c in candidates:  # pad with lexical order if the model returned too few
            if len(picked) >= top_k:
                break
            if c not in picked:
                picked.append(c)
        return picked[:top_k]
