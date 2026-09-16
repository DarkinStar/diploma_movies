"""Cross-encoder reranking of dense-retrieval candidates.

Status: evaluated and NOT used in the final system.

A cross-encoder (cross-encoder/ms-marco-MiniLM-L6-v2 fine-tuned on 8,040
query/plot pairs) was applied to the top-10 candidates of the LoRA v2 retriever.
On the 394-query test set it dropped Hit@1 from 39.6% to 10.7%
(results/eval_394_lora_v2_cross_encoder.csv). The most likely cause is that the
candidates passed to the reranker carried only titles, so the reranker scored
"query vs. title" while it had been trained on "query vs. plot". See
docs/experiments.md. The module is kept so the experiment stays reproducible.
"""
from pathlib import Path

from sentence_transformers import CrossEncoder

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "cross_encoder"

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(str(MODEL_PATH))
    return _reranker


def rerank(query: str, candidates: list[dict], text_field: str = "summary") -> list[dict]:
    """Re-order `candidates` (dicts with at least "title") by cross-encoder score.

    `text_field` selects which candidate field is concatenated to the title;
    it should be the same kind of text the cross-encoder was trained on.
    """
    pairs = [[query, f"{c['title']}. {c.get(text_field) or ''}".strip()] for c in candidates]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [candidate for _, candidate in ranked]
