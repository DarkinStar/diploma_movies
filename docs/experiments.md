# Experiment log

Everything below was run on a laptop CPU. Numbers are on the 394-query test set unless stated otherwise; the per-query CSVs are in `results/`.

## Setup

- **Corpus:** 32,364 movies from the Kaggle *Wikipedia Movie Plots with Plot Summaries* dump after two cleaning passes (notebook 01, `scripts/clean_dataset.py`). One composite document per movie: `Title: … Genre: … Summary: … Plot: …`.
- **Test set:** 98 movies × 4 LLM-generated query types (`oracle`, `naturalistic`, `conversational`, `vague`), 394 queries in total (`data/eval/test_set_394.csv`). A separate 90-query set (30 × 3 types) was used during development and is not reported.
- **Metrics:** top-10 retrieval; the correct title is matched fuzzily (`difflib.SequenceMatcher` ratio ≥ 0.85) so that punctuation and "(film)" suffixes do not count as misses. Hit@1/5/10 and MRR are averaged over queries.
- **Models:** `mixedbread-ai/mxbai-embed-large-v1` (BERT-large, 335 M parameters, 1024-d) as the embedding model; queries are prefixed with `Represent this sentence for searching relevant passages:` as the model card recommends. Exact search with `faiss.IndexFlatIP` over L2-normalised vectors.

## Timeline

| When | What | Result (Hit@1 / Hit@10 / MRR) |
|---|---|---|
| May 2025 | First prototype: `all-MiniLM-L6-v2` (384-d) index over the first cleaning pass, FastAPI + ngrok demo. | not formally evaluated |
| 9 Apr 2026 | Second cleaning pass (`clean_dataset.py`), switch to `mxbai-embed-large-v1`, base index. | 37.3 % / 52.0 % / 0.415 |
| 12–13 Apr | ~5 K synthetic queries with Llama-3.1-8B (Groq). Three full fine-tune runs of all weights (`MultipleNegativesRankingLoss`, batch 4, 1–3 epochs). Every run scored below the base model on the 90-query dev set; the last one was evaluated on the full test set. | 11.2 % / 37.8 % / 0.172 |
| 15 Apr | Hard-negative mining (notebook 05): top-5 base-model candidates re-ranked by the LLM, bottom three kept as negatives (2,010 rows). | – |
| 15 Apr | LoRA adapters (notebook 06): rank 16 on q/k/v, 500 pairs, batch 4, 1 epoch. A first adapter trained on 1,608 pairs with batch 8 was not kept. v1 and v2 differ only in the sampled pairs. | v1: 39.6 % / 56.3 % / 0.444, v2: 39.6 % / 56.6 % / 0.445 |
| 15 Apr | Cross-encoder reranker: `cross-encoder/ms-marco-MiniLM-L6-v2` fine-tuned on 8,040 query/plot pairs (binary cross-entropy, 1 epoch), applied to LoRA v2's top-10. | 10.7 % / 37.3 % / 0.171 |
| 4 Jun | BM25 baseline (`rank_bm25`, stop-word removal) on the same composite documents. | 37.6 % / 49.5 % / 0.412 |

The final system is **LoRA v2 without the reranker**.

## Hit@1 / Hit@10 by query type

| Model | oracle (100) | naturalistic (98) | conversational (98) | vague (98) |
|---|---|---|---|---|
| BM25 | 100 % / 100 % | 28 % / 50 % | 20 % / 43 % | 1 % / 4 % |
| mxbai, no fine-tuning | 74 % / 91 % | 32 % / 51 % | 39 % / 58 % | 4 % / 7 % |
| Full fine-tune | 18 % / 63 % | 12 % / 38 % | 13 % / 45 % | 1 % / 5 % |
| LoRA v2 | 81 % / 96 % | 35 % / 58 % | 38 % / 63 % | 4 % / 8 % |
| LoRA v2 + reranker | 18 % / 63 % | 10 % / 38 % | 13 % / 45 % | 1 % / 3 % |

## Why the full fine-tune collapsed

Observed: Hit@1 37.3 % → 11.2 %; `oracle` queries 74 % → 18 %. Queries that literally quote the plot should be the easiest case, so losing them means the model's general text matching degraded, not just its handling of the new query style.

Contributing factors, in the order I believe matters most:

1. **All 335 M weights trainable with batch size 4.** `MultipleNegativesRankingLoss` uses the other items in the batch as negatives; with batch 4 that is three negatives per step, a very weak signal to move a large model with. The LoRA runs use the same loss and batch size but touch only 2.4 M parameters, and did not collapse.
2. **Only synthetic queries as supervision.** Every training query came from one LLM prompt template, so the model could fit the template's style rather than the retrieval task.
3. **Truncated documents.** Plots were cut to 256 characters and the sequence length to 128 tokens to fit into memory, so the model was trained on document fragments but evaluated on full composite documents.

## Why the reranker collapsed

Observed: applied to LoRA v2's top-10, Hit@1 39.6 % → 10.7 %; 120 of the 156 queries that LoRA v2 had at rank 1 were pushed down. The retriever's candidate set was unchanged, so the damage is entirely in the reordering.

Most likely cause: the candidate dictionaries returned by the search function carried only `title` (the same shape the API returned at the time), and the reranker concatenated `title + plot` with `plot` defaulting to an empty string. The cross-encoder therefore scored *query vs. title* at inference after being trained on *query vs. plot*. A secondary factor is the training data: 8 K pairs with LLM-generated labels and no hard negatives is little for a cross-encoder. The rerank function now takes the text field explicitly (`src/reranker.py`); re-running the experiment with plots or summaries is the obvious next step.

## Limitations and next steps

- **Vague queries are unsolved** by every method (≤ 4 % Hit@1). They may need query expansion or a conversational interface rather than a better encoder.
- **Test queries are LLM-generated**, as are the training queries, so the gains of fine-tuning may partly reflect matching the generator's style. A hand-written or user-collected query set would be a stronger test.
- **Train/test overlap is small but not zero.** 2 of the 98 test movies (8 queries) also appear as positives in the LoRA training pairs, with different queries. Excluding them, LoRA v2 still improves Hit@1 from 37.3 % to 39.4 % on the remaining 386 queries.
- **LoRA was trained on 500 pairs for one epoch** because of the CPU budget. The mined hard negatives (notebook 05) were never used as explicit negatives; a triplet or `MultipleNegativesRankingLoss`-with-hard-negatives run on a GPU is the natural continuation.
- **Fuzzy title matching** (ratio ≥ 0.85) can in principle accept a wrong title that is very similar to the right one (sequels, remakes); a check by row id would be stricter.
- **Exact FAISS search** is fine for 32 K documents (a query takes milliseconds); an IVF or HNSW index would be needed well before millions of documents.

## Files in `results/`

One CSV per run, one row per test query. Columns: `title`, `year`, `genre`, `query`, `query_type`, `top1_result`, `correct_rank` (1–10 or `not found`), `match_ratio`, `hit@1`, `hit@3` (some runs), `hit@5`, `hit@10`, `reciprocal_rank`, `similarity_score` (or `bm25_score`).

| File | Run |
|---|---|
| `eval_394_bm25.csv` | BM25 baseline (notebook 09) |
| `eval_394_mxbai_base.csv` | base model (notebook 07) |
| `eval_394_full_finetune.csv` | full fine-tune (notebook 04) |
| `eval_394_lora_v1.csv`, `eval_394_lora_v2.csv` | LoRA adapters (notebook 08) |
| `eval_394_lora_v2_cross_encoder.csv` | LoRA v2 top-10 re-ordered by the cross-encoder |

`python scripts/summarize_results.py` prints the tables in the README from these files.
