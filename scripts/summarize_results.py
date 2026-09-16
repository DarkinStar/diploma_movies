"""Print the evaluation tables shown in the README from the per-query CSVs in results/.

Usage (from the repository root):
    python scripts/summarize_results.py
"""
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# (file, label) in the order the rows should appear.
RUNS = [
    ("eval_394_bm25.csv", "BM25 (keyword baseline)"),
    ("eval_394_mxbai_base.csv", "mxbai-embed-large-v1, no fine-tuning"),
    ("eval_394_full_finetune.csv", "Full fine-tune (all weights)"),
    ("eval_394_lora_v1.csv", "LoRA v1"),
    ("eval_394_lora_v2.csv", "LoRA v2 (final system)"),
    ("eval_394_lora_v2_cross_encoder.csv", "LoRA v2 + cross-encoder reranker"),
]
QUERY_TYPES = ["oracle", "naturalistic", "conversational", "vague"]


def pct(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    frames = {label: pd.read_csv(RESULTS_DIR / name) for name, label in RUNS}

    print("### Overall (394 queries, 98 movies, top-10 retrieval)\n")
    print("| Model | Hit@1 | Hit@5 | Hit@10 | MRR |")
    print("|---|---|---|---|---|")
    for label, df in frames.items():
        print(
            f"| {label} | {pct(df['hit@1'].mean())} | {pct(df['hit@5'].mean())} | "
            f"{pct(df['hit@10'].mean())} | {df['reciprocal_rank'].mean():.3f} |"
        )

    print("\n### Hit@1 / Hit@10 by query type\n")
    print("| Model | " + " | ".join(QUERY_TYPES) + " |")
    print("|---|" + "---|" * len(QUERY_TYPES))
    for label, df in frames.items():
        cells = []
        for query_type in QUERY_TYPES:
            sub = df[df["query_type"] == query_type]
            cells.append(f"{sub['hit@1'].mean():.0%} / {sub['hit@10'].mean():.0%}")
        print(f"| {label} | " + " | ".join(cells) + " |")

    counts = frames[RUNS[0][1]]["query_type"].value_counts()
    print("\nQueries per type: " + ", ".join(f"{t}={counts.get(t, 0)}" for t in QUERY_TYPES))


if __name__ == "__main__":
    main()
