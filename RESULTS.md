# RESULTS — flat experiment table

One row per measured change. `PR` = pair recall, `FCR` = full-cluster recall,
`cand/S1` = average candidate-list size, `ceiling` = the best macro F_0.5 any matcher
could reach on that candidate set. Validation F_0.5 columns are macro over all S1
entities in the fold, singletons included. `Reweighted` = test-mix weighted
(0.4675·India + 0.3827·US + 0.1498·LOCO, derived from `test_source1.tsv`).

All blocking rows below: **train / India, 4,000 held-out fold-0 queries searching the
whole 4,133,346-record S2+S3 pool**, top-k per source.

## Blocking — representation

| date | change | k | PR | FCR | cand/S1 | ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-25 | char-3 grams, name+address (the doc's recipe) | 30 | 0.8335 | 0.6354 | 60 | — |
| 2026-09-25 | char-5 grams | 50 | 0.8499 | 0.6639 | 100 | — |
| 2026-09-25 | word + char-4, name+address | 20 | 0.8908 | 0.7369 | 40.0 | 0.9543 |
| 2026-09-25 | word + char-4, address only | 20 | 0.8183 | 0.5794 | 40.0 | 0.9253 |
| 2026-09-25 | word + char-4, name only | 20 | 0.5584 | 0.2953 | 40.0 | 0.7151 |
| 2026-09-25 | char-3 on consonant skeleton | 20 | 0.8287 | 0.6093 | 39.9 | 0.9241 |
| 2026-09-25 | char-3 on skeleton of name only | 20 | 0.3942 | 0.1842 | 38.4 | 0.5442 |

## Blocking — unions (the four views fail independently)

| date | change | k | PR | FCR | cand/S1 | ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-25 | nameaddr | 20 | 0.8908 | 0.7369 | 40.0 | 0.9543 |
| 2026-09-25 | nameaddr + skeleton | 20 | 0.9184 | 0.7905 | 69.1 | 0.9678 |
| 2026-09-25 | nameaddr + name + addr | 20 | 0.9317 | 0.8252 | 97.3 | 0.9732 |
| 2026-09-25 | **nameaddr + name + addr + skeleton** | 20 | **0.9468** | **0.8550** | 124.9 | **0.9801** |
| 2026-09-25 | the same four | 30 | 0.9541 | 0.8729 | 189.4 | 0.9832 |
| 2026-09-25 | + skeleton-of-name (dropped) | 20 | 0.9491 | 0.8603 | 153.9 | 0.9812 |

## Blocking — cost knobs (name+address view)

| date | change | df cap | budget | PR | FCR | s / 1k queries |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-25 | df cap sweep | 12,000 | 15,000 | 0.8303 | 0.6330 | 0.3 |
| 2026-09-25 | | 12,000 | 120,000 | 0.8729 | 0.6983 | 1.5 |
| 2026-09-25 | | 12,000 | 240,000 | 0.8746 | 0.7034 | 1.6 |
| 2026-09-25 | | 40,000 | 30,000 | 0.8482 | 0.6671 | 0.6 |
| 2026-09-25 | | 40,000 | 120,000 | 0.8809 | 0.7207 | 1.9 |
| 2026-09-25 | | 40,000 | 240,000 | 0.8949 | 0.7468 | 3.7 |
| 2026-09-25 | **skeleton view under the same absolute cap** | 12,000 | 120,000 | **0.5900** | 0.3510 | 0.4 |
| 2026-09-25 | skeleton view, fractional cap | 6% of pool | 200,000 | 0.8287 | 0.6093 | 3.0 |

The last two rows are why the df cap is **per retriever**: the skeleton view has only
~11k possible 3-grams, so an absolute 12k-document cap deletes its vocabulary.

## Blocking — production configuration

`b3_k30_u80_df1_qb60k_key20_pr60`: four retrievers at top-30 per source, union capped
at 80 per source, per-view df caps (20k documents for the word/char views, 3% of the
pool for the skeleton), query budget 60k postings, key blocks capped at 50 records,
pre-ranker keeping 15–60 per S1 above eps=0.002.

Measured throughput, India train S2 (2,017,799 documents, top-30):

| retriever | s / 1k queries | index nnz | index size |
| --- | --- | --- | --- |
| name+address | 0.33 | 58.6M | 0.47 GB |
| name only | 0.18 | 21.4M | 0.17 GB |
| address only | 0.33 | 43.9M | 0.35 GB |
| skeleton | 0.89 | 25.4M | 0.20 GB |
| **all four** | **1.73** | | |

## Blocking, full run — cost per partition

Four retrievers at top-30 per source, union capped at 80 per source. Query rate scales
with pool size, which is why France (700k records per source) is four times cheaper per
query than India (2.3M).

| split | country | queries | source | candidate pairs | seconds | s / 1k queries |
| --- | --- | --- | --- | --- | --- | --- |
| train | India | 370,000 | S2 | 29,555,832 | 681 | 1.84 |
| train | India | 370,000 | S3 | 29,562,111 | 797 | 2.15 |
| train | US | 370,001 | S2 | 29,579,040 | 962 | 2.60 |
| train | US | 370,001 | S3 | 29,580,812 | 1,048 | 2.83 |
| test | France | 259,452 | S2 | 20,678,849 | 162 | 0.62 |
| test | France | 259,452 | S3 | 20,681,844 | 173 | 0.67 |
| test | India | 809,986 | S2 | 64,695,074 | 2,049 | 2.53 |
| test | India | 809,986 | S3 | 64,704,753 | 1,906 | 2.35 |

## Pre-ranker pruning: ceiling vs candidates per S1

Held-out fold 0, full-pool retrieval, 4,000 queries per country.

| max / min / eps | India cand/S1 | India ceiling | US cand/S1 | US ceiling |
| --- | --- | --- | --- | --- |
| 20 / 8 / 0.05 | 8.4 | 0.9437 | 8.4 | 0.9825 |
| 30 / 10 / 0.02 | 10.7 | 0.9474 | 10.6 | 0.9848 |
| 40 / 12 / 0.01 | 13.1 | 0.9500 | 13.0 | 0.9864 |
| 50 / 12 / 0.005 | 16.4 | 0.9547 | 15.1 | 0.9876 |
| **60 / 15 / 0.002** | **28.4** | **0.9609** | **25.3** | **0.9907** |
| 80 / 15 / 0.0 | 80.0 | 0.9681 | 80.0 | 0.9930 |
| raw union | 118.9 | 0.9714 | 119.1 | 0.9937 |

## Engineering changes that changed the numbers

| date | change | effect |
| --- | --- | --- |
| 2026-09-25 | keep LightGBM's OpenMP runtime out of the retrieval process | **130 -> 3,351 queries/s (25x)**; projected blocking 16 h -> 1.6 h |
| 2026-09-25 | free the pool text before the query loop | ~0.6 GB less resident per partition |
| 2026-09-25 | give the pre-ranker its own fold | removes an in-sample feature that was 38.6% of M1's gain |
| 2026-09-25 | df cap 40k -> 20k documents (word/char views) | pair recall 0.8809 -> 0.8729, index nnz roughly halved |
| 2026-09-25 | query budget 120k -> 60k postings | pair recall 0.8809 -> 0.8642 alone; less in the union |
| 2026-09-25 | index cache key carries the caps, not just a scale | fixes silently reusing an index built with different settings |
| 2026-09-25 | filter feature shards before stacking | avoids loading a full country matrix to discard most of it |
| 2026-09-25 | writer streams numpy arrays instead of a dict of strings | ~4 GB -> ~0.5 GB at test scale |

## Matching

| date | change | val F0.5 US | val F0.5 India | LOCO | reweighted |
| --- | --- | --- | --- | --- | --- |
| 2026-09-25 | Phase 0, 4k S1/country training sample (proving run) | 0.9495 | 0.8982 | n/a | 0.9213 |
| 2026-09-25 | full run, feature set v1 | 0.9571 | 0.9151 | 0.8904 | 0.9275 |
| 2026-09-25 | feature set v2 (one-sided token counts, full-name skeleton) | 0.9591 | 0.9246 | 0.8860 | 0.9320 |
| 2026-09-26 | **v3**: 10 pair + 6 meta columns, `eps` 0.002 → 0.001 | 0.9610 | 0.9321 | 0.8895 | 0.9368 |
| 2026-09-26 | **v4**: + selectivity-ordered key blocks, cap 150 | **0.9613** | **0.9340** | **0.8906** | **0.9380** |

Per-country tuned rules. Under the single rule the submission actually applies to all three
countries, v3 reads 0.9362 against v2's 0.9320.

## Pruning threshold — `eps` binds, `prerank_max` does not

Cached raw union, held-out fold 0, full-pool retrieval. `max` at 60 / 80 / 120 gives
**bit-identical** candidate sets: the average list is far shorter than the cap, so the
threshold is what decides its length.

| max / min / eps | India cand/S1 | India PR | India FCR | India ceiling | US ceiling |
| --- | --- | --- | --- | --- | --- |
| 60 / 15 / 0.002 (submitted) | 29.4 | 0.9003 | 0.7492 | 0.9595 | 0.9861 |
| 120 / 15 / 0.002 | 29.4 | 0.9003 | 0.7492 | 0.9595 | 0.9861 |
| **120 / 15 / 0.001 (adopted)** | 47.2 | 0.9090 | 0.7687 | **0.9632** | **0.9878** |
| 120 / 15 / 0.0005 | 81.6 | 0.9181 | 0.7893 | 0.9672 | 0.9893 |
| raw union, no pruning | 159.8 | 0.9282 | 0.8133 | 0.9715 | 0.9909 |

## Key blocks — ordering the hits by selectivity, and a fifth retriever

4,000 held-out fold-0 India queries against the whole 4,133,346-record S2+S3 pool, union
capped at 80 per source, so every row costs the same number of candidates.

| combination | cand/S1 | PR | FCR | ceiling |
| --- | --- | --- | --- | --- |
| 4 views, no keys | 159.2 | 0.9196 | 0.7955 | 0.9656 |
| 4 views + keys, pool order, cap 50 | 159.8 | 0.9209 | 0.7979 | 0.9674 |
| **4 views + keys, selective, cap 150** | 159.7 | **0.9361** | **0.8279** | **0.9747** |
| 4 views + LSH + keys, pool order | 160.0 | 0.9120 | 0.7756 | 0.9630 |
| 4 views + LSH + keys, selective | 160.0 | 0.9271 | 0.8048 | 0.9703 |
| banded MinHash (LSH) alone | 59.8 | 0.7348 | 0.4217 | 0.8770 |

**Selectivity ordering: +0.0073 of ceiling at an identical candidate count and identical
cost.** Hits used to come back in pool order and the caller kept the first twenty with
`np.unique`, which *sorts* — so a query matching a 3-record key and a 140-record key kept
twenty rows chosen by lowest pool index. It is also why raising the cap alone would have
hurt.

**MinHash / LSH: rejected.** It lowers the ceiling. The argument for it — LSH is rank-free,
so a true match at cosine rank 200 still collides — is sound in isolation and wrong at a
fixed candidate budget: a fifth view displaces candidates rather than adding them, and LSH
alone reaches 0.7348 pair recall against name+address's 0.89. It is cheap and correct
(75,668 records/s to sign, **0.04 s/1k queries**, sixteen times less than the skeleton
view) and stays in the tree behind its config, off.

## Decision layer — exhausted

180 configurations on the v2 stack: 9 temperatures x 10 candidate-miss probabilities x 2
countries, reusing cached scores.

| | India | US | reweighted (India+US) |
| --- | --- | --- | --- |
| baseline `T=0.6, miss=0` | 0.92464 | 0.95907 | 0.94014 |
| best of all 180 | 0.92477 | 0.95909 | **0.94016** |

**+0.00002.** India spans 0.9239–0.9248 across every temperature from 0.35 to 1.25. This
also bounds exact F-measure maximisation under dependency: the whole decision-rule family
spans ~0.002 here, so a better solver inside it cannot be worth more than that. With the
longer v3/v4 candidate list every fold does select a non-zero `miss_prob`, so the knob pays
only in combination.

## Phase 1 — dense bi-encoder, code-proving run

A deliberately tiny slice (404 queries, 4,000-record pool, **4 training steps**) whose only
purpose was to prove the Phase-1 path runs end to end on Apple MPS — build the slice, run
the sparse baseline in one process, then encode, fine-tune with SupCon on collision
batches and re-encode in another. The slice is far too easy to draw conclusions from; the
one thing it does show is that **training moves the encoder in the right direction**.

| retriever | pair recall | full-cluster recall | cand/S1 |
| --- | --- | --- | --- |
| sparse union (4 views) | 0.9941 | 0.9845 | 72.3 |
| dense, zero-shot (multilingual-e5-small) | 0.9522 | 0.9093 | 30.0 |
| dense, after 4 SupCon steps | **0.9780** | **0.9508** | 30.0 |
| sparse ∪ dense | 0.9985 | 0.9974 | 92.3 |

Encoder throughput on MPS: **357–380 records/s** (multilingual-e5-small, 384-d, 64 tokens).
Extrapolated to BGE-M3 over 22M records on a 24 GB GPU this is consistent with the
research doc's 3–5 h estimate for a full encode.
