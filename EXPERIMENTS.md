# EXPERIMENTS — branch `Experiment`

Log for the recall push on top of the submitted phase-0 pipeline. One section per
change: what it was meant to do, what was measured, and whether it was kept.

**Branch:** `Experiment` (created off `main` at `ef4b880`).
`main` holds the submitted pipeline; nothing here is merged until it beats it on
the held-out fold.

**Goal given:** lift the public-leaderboard score from 0.92 towards 0.98.
See [§7](#7-can-this-reach-098) for what is actually reachable on this machine
and what is not — the honest answer is **no, not from here**, and the reason is a
ceiling that no amount of feature work moves.

---

## 0. First: the baseline in `results/` is stale

`results/BENCHMARK.md` and `APPROACH.md` report **India 0.9151 / reweighted
0.9275**. Those are the **v1** feature-set numbers. A second full run with the v2
features finished afterwards (`work/reports/full_run_v2.log`, 00:37) and was never
folded into `results/`. The real baseline this branch has to beat is:

| | India | US | LOCO mean | reweighted |
| --- | --- | --- | --- | --- |
| v1 (what `results/` says) | 0.9151 | 0.9571 | 0.8904 | 0.9275 |
| **v2 (what the code actually does)** | **0.9246** | **0.9591** | **0.8860** | **0.9320** |

Config `b3_k30_u80_df1_qb60k_key20_pr60`, feature set v2, rule `('eum', 0.6, 0.0)`.
India P/R = 0.983 / 0.842, US P/R = 0.988 / 0.912.

Everything below is measured against the **v2** row.

### Where the loss actually is

| | achieved | candidate ceiling | matcher gap | blocking gap |
| --- | --- | --- | --- | --- |
| India | 0.9246 | 0.9601 | **0.0355** | 0.0399 |
| US | 0.9591 | 0.9866 | **0.0275** | 0.0134 |

So the matcher's own gap is the larger half for both countries — not the
two-thirds-is-blocking split `APPROACH.md` claims from the v1 run. And at
P = 0.983 / R = 0.842 the metric's marginal rates are
`dlnF/dlnR = 0.82` against `dlnF/dlnP = 0.18`: **one point of recall is worth
about 4.6 points of precision here**, so every change below is judged on whether
it buys recall without collapsing precision.

---

## 1. Measured before implementing — two ideas killed in 20 minutes

Both probes reuse the cached v2 stack and v2 feature shards. No re-blocking, no
re-featurising, no re-training, so each cost one scoring pass.

### 1.1 The decision layer is exhausted — REJECTED

Probe: `work/reports/probe_decision.json` (log `probe_decision.log`, 1188 s).
Swept 9 temperatures × 10 candidate-miss probabilities × 2 countries = 180
configurations on the held-out fold.

The hypothesis was strong on paper. The production rule maximises expected F_0.5
assuming the candidate list is **complete**, and India's pruned full-cluster
recall is 0.754 — for a quarter of entities at least one true match never reaches
the matcher. Telling the rule that (`miss_prob > 0`) raises the true-set size it
budgets for, which makes one more predicted candidate cheaper and the empty
prediction worthless. The knob was already implemented and **never swept**
(`miss_probs=(0.0,)`). The first run also chose `T = 0.6`, the *edge* of the old
temperature grid, which usually means the optimum is outside it.

| | India | US | reweighted (India+US) |
| --- | --- | --- | --- |
| baseline `T=0.6, miss=0` | 0.92464 | 0.95907 | 0.94014 |
| best of all 180 (`T=0.8, miss=0.4`) | 0.92477 | 0.95899 | **0.94016** |

**+0.00002.** The whole grid is flat to four decimals: India spans
0.9239–0.9248 across every temperature from 0.35 to 1.25, and `miss_prob` moves
recall by 0.003 at most (0.8467 → 0.8534 from `miss=0` to `miss=0.5`) while
precision gives back the same. The rule is insensitive because the probabilities
are sharp: once most candidates sit near 0 or 1, the arg-max over prefix length
is decided by the probability gap, not by the denominator that `miss_prob` shifts.

**This also rejects the two proposals aimed at this layer:**

* **Dembczyński's exact F-measure maximisation under dependency** — the
  independence assumption is not what is costing anything. The entire
  decision-rule family spans ~0.002 on this data (tuned threshold vs expected-F,
  every temperature, every miss probability), so a better solver *inside* that
  family cannot be worth more than that. It would also need a joint distribution
  `P(y)` from a CRF/MRF that does not exist and would need its own model and
  training fold; the O(N³) solver is the easy part. And the dominant real
  dependency — two S1 entities claiming the same record — is already handled in
  closed form by the exclusivity correction, which M2 confirms is doing the work
  (`m_excl_prob` was 61.5% of its gain).
* **"Tune the calibration holdout"** — already representative. Fold 4b is
  family-sampled from the same distribution as 4a, and ECE is 0.0003 → 0.0000
  after isotonic regression. There is no skew to fix.

**Kept anyway, because it is free:** the grid is now widened
(`DECISION_TEMPS`, `DECISION_MISS` in `pipeline.py`) and the sweep was made ~9×
cheaper. `expected_f05_components` splits E[F] into the two halves that
`miss_prob` mixes — neither depends on it — so a whole sweep costs one pass per
temperature instead of one per (temperature, miss) pair. Exactly the same
arithmetic; `tests/test_decide.py` still passes unchanged.

### 1.2 `prerank_max` was never the binding constraint — the proposal was wrong, the *diagnosis* was right

Probe: `work/reports/probe_prune.json`. Scores the cached raw union with the
cached pre-ranker and recomputes the candidate ceiling for a grid of thresholds,
in numpy from labels and cluster sizes rather than from dicts of id strings —
which is what makes a 19M-row table cost 3 seconds.

The proposal was to raise the pre-ranker's ceiling from 60 to 100–120 per entity
so large corporate parks and malls keep their candidates. Measured, India,
held-out fold 0, full-pool retrieval:

| max / min / eps | cand/S1 | pair recall | full-cluster recall | F₀.₅ ceiling |
| --- | --- | --- | --- | --- |
| **60 / 15 / 0.002** (the submitted run) | 29.4 | 0.9003 | 0.7492 | 0.9595 |
| 80 / 15 / 0.002 | 29.4 | 0.9003 | 0.7492 | 0.9595 |
| 120 / 15 / 0.002 | 29.4 | 0.9003 | 0.7492 | 0.9595 |
| 120 / 15 / 0.001 | 47.2 | 0.9090 | 0.7687 | 0.9632 |
| 120 / 15 / 0.0005 | 81.6 | 0.9181 | 0.7893 | 0.9672 |
| 160 / 20 / 0.0002 | 141.7 | 0.9269 | 0.8099 | 0.9710 |
| raw union (no pruning) | 159.8 | 0.9282 | 0.8133 | 0.9715 |

Raising `max` from 60 to 120 is **bit-identical**. It never binds: the average
list is 29 candidates, so the threshold `eps` is what decides the length and
`max` only clips a tail that barely exists. US is the same shape
(25.2 cand/S1, ceiling 0.9861, unchanged by `max`).

**What is true is the underlying claim** — the pruning threshold, not the model,
is discarding 0.012 of India's ceiling — and `eps` is the knob that recovers it.
Adopted: **`eps` 0.002 → 0.001**, `max` → 120 (harmless, and it stops binding
if `eps` drops further). That is +0.0037 of ceiling for 1.6× the pairs, which is
what the per-pair stages of a run cost. `eps = 0.0005` buys +0.0077 for 2.8× and
is the obvious next step once the matcher is shown to capture this one.

---

## 2. What was implemented

### 2.1 Ten new pair features (v3), aimed at the measured error buckets

The false-negative table is the brief. India's top buckets are *name differs by a
legal form or one token* (22.3%), *no shared name token* (16.0%), *missing address
on one side* (15.7%), *no house number on one side* (14.3%), *native-script name*
(10.6%). For the US, *missing address on one side* alone is **26.6%**.

| feature | what it sees that nothing else did |
| --- | --- |
| `xf_n1_a2`, `xf_n2_a1` | **cross-field leakage.** The business name written into the *address* line, or the reverse. Every name feature compares name to name and every address feature address to address, so this pair reads as two conflicts instead of one match. Measured on a true training pair whose *only* shared evidence is an S1 **name** token appearing in the candidate's **address**. |
| `name_idf_jac` | IDF-weighted Jaccard on the **full** token sets. The existing IDF features are containment and unmatched mass over the *core* name — which strips exactly the tokens the dataset's hard negatives are built from. |
| `name_rare_shared` | **was anything rare shared**, normalised by the country's max IDF. Distinct from "how much weight is shared": one rare token in common decides a pair whose address is missing, and a dozen common ones do not. |
| `name_skel_lev` | edit distance on the consonant skeleton. `name_skel_jac` is gram-based and therefore order-free, so a fully transposed spelling scores like a near-identical one. |
| `adm_tail_jac`, `adm_tail_cov` | **manufactured administrative geography.** External geocoding is banned, so the region comes out of the string: the trailing non-numeric tokens are the city/state in every country here. `_cov` compares them against the *whole* other address because these fields get reordered ("OH, Columbus, 5559 Kingsmere Avenue"), where a strict tail-to-tail test compares a street against a city. |
| `adm_pin_match` | a shared 5–6 digit token — ZIP and PIN alike, **by length only**, so no country is named. Three-valued: match / conflict / NaN, because no PIN is not a conflict. |
| `addr_skel_jac` | the consonant skeleton of the **address**, for the reason the name already has one: a Devanagari address shares no token with its romanisation, so every other address feature reads a conflict. |
| `addr_rare_shared` | a shared rare street or locality token, which is evidence even when the house numbers disagree — the second- and fourth-largest US false-negative buckets. |

All ten carry a monotone constraint, and the two rarity columns are divided by
the country's IDF ceiling so they mean the same thing in a country the model never
saw — which is what decides the 15%-weighted LOCO term.

### 2.2 Six new meta features — the shape of the candidate list

`m_cliff` (relative step-drop to the runner-up), `m_gap_rel`, `m_std`, `m_psum`
(the list's own estimate of its cluster size), `m_rank_src`, `m_excl_loss`.

A crowded list is not evidence against the candidate at the top of it. Without
these, M2 sees "this candidate has eight close competitors" and cannot tell a
genuine ambiguity from a chain of identically named branches. `m_rank_src` is the
one that was plainly missing: the metric counts S2 and S3 separately, so a
candidate's real competition is its own source, and `m_rank` mixes the two.

*Triadic closure / cross-source support was already in the pipeline*
(`x_support_name`, `x_support_addr`). It was anchored on the RRF-best candidate
of the other source; it is now anchored on the **pre-ranker's** best, which is a
model trained to rank rather than an unweighted rank sum.

### 2.3 Two cache-correctness fixes the sweeps forced

* **Retrieval and pruning fingerprints are now separate.** The raw union of one
  country costs an hour of retrieval and 5 GB on disk, and it does not depend on
  how it is later pruned — but `cand_dir` keyed both on one fingerprint, so
  sweeping a pruning threshold re-ran blocking for nothing. Raw shards now key on
  `retrieval_fingerprint()`, pruned shards and features on the full one. This is
  what let the whole sweep above happen on cached data.
* **`prerank_min` and `prerank_eps` are now *in* the fingerprint.** They were
  not: `_pr60` named only `max`. Two configurations differing only in `eps`
  produce different candidate sets and would have silently shared shards — the
  same class of bug as the two recorded in `PROGRESS.md` §7.

### 2.4 Key blocks: the per-query cut was arbitrary (off by default)

`KeyBlocks.lookup` returned hits in pool order and the caller kept the first 20
via `np.unique`, **which sorts** — so a query hitting a 3-record key and a
140-record key kept twenty rows chosen by lowest pool index, not the three
precise ones. That is also why raising the block cap on its own would have made
recall *worse*: a bigger cap crowds a precise hit out with the low-numbered rows
of a vague one. Hits are now ordered by block size and deduped order-preservingly.

**Off by default (`key_select=False`).** It changes the raw union, and the 5 GB
of cached unions were built with the old behaviour; a cached artefact must never
mean two things. Turning it on appends `_ks` to the retrieval fingerprint, so it
gets its own namespace and has to earn a re-block on the benchmark.

---

## 3. Results

Baseline is the v2 row from §0. Config
`b3_k30_u80_df1_qb60k_key20_pr120-15-0.001`, feature set v3.

### Candidate ceiling (held-out fold 0, full-pool retrieval)

| run | India cand/S1 | India PR | India FCR | India ceiling | US cand/S1 | US ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| v2 baseline | 33.8 | 0.9021 | 0.7537 | 0.9601 | 31.6 | 0.9866 |
| **v3 (`eps=0.001`)** | 47.6 | **0.9091** | **0.7690** | **0.9632** | 42.0 | **0.9878** |

The pruning probe predicted 0.9632 and 0.9878 from the cached raw union; the run
reproduced both exactly, which is the point of having measured it first.

### Matching (held-out fold 0, 120,000 entities per country)

| run | India | US | LOCO India→US | LOCO US→India | LOCO mean | reweighted |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | 0.9151 | 0.9571 | — | — | 0.8904 | 0.9275 |
| v2 (baseline) | 0.9246 | 0.9591 | 0.9354 | 0.8366 | 0.8860 | 0.9320 |
| **v3** | **0.9321** | **0.9610** | **0.9365** | **0.8424** | **0.8895** | **0.9368** |
| Δ vs v2 | **+0.0075** | +0.0019 | +0.0011 | **+0.0058** | +0.0034 | **+0.0048** |

Precision and recall both rose in both countries — India 0.983/0.842 → 0.985/0.857,
US 0.988/0.912 → 0.990/0.914 — so this is not a precision-for-recall trade.

**Where the gain comes from.** India's ceiling moved +0.0031 and India's score moved
+0.0075, so the matcher's *capture rate* against its own ceiling improved from
96.30% to 96.77%. Roughly 0.003 of the gain is the longer candidate list and
roughly 0.004 is the new features converting headroom that was already there.

**The France proxy improved second-most** (+0.0058 on US→India). That is the
result the two rarity columns were designed for: both are divided by the
country's own maximum IDF so "a rare word agreed" means the same thing in a
country whose corpus is a different size, and leave-one-country-out is the only
honest test of it.

**The widened decision grid is now used.** Both countries selected a non-zero
`miss_prob` (0.4 India, 0.2 US) where the v2 candidate set had no use for one
(§1.1 measured +0.00002). The knob only pays off in combination with a longer
candidate list — on its own it is worthless, which is why it is listed under both
"rejected" and "kept".

### Which of the new columns earned their place

Share of total LightGBM gain, and rank among all of that model's features.

| column | M1 gain | M1 rank | verdict |
| --- | --- | --- | --- |
| `name_skel_lev` | 2.24% | **7 / 62** | clear winner |
| `addr_skel_jac` | 0.94% | 15 / 62 | earns its place |
| `name_idf_jac` | 0.92% | 16 / 62 | earns its place |
| `addr_rare_shared` | 0.15% | 30 / 62 | marginal |
| `name_rare_shared` | 0.08% | 38 / 62 | marginal |
| `adm_pin_match` | 0.01% | 49 / 62 | ~nothing |
| `adm_tail_cov` | **0.00%** | 51 / 62 | **dead** |
| `xf_n1_a2` | **0.00%** | 53 / 62 | **dead** |
| `adm_tail_jac` | **0.00%** | 55 / 62 | **dead** |
| `xf_n2_a1` | **0.00%** | 60 / 62 | **dead** |

The ten together are 4.34% of M1's gain. All six meta columns contribute:
`m_gap_rel` 4.10% (**rank 3 / 77**), `m_cliff` 0.20%, `m_std` 0.18%, `m_psum`
0.16%, `m_excl_loss` 0.07%, `m_rank_src` 0.06% — 4.85% of M2's gain together.

**Two of the proposed ideas are therefore measured dead.**

* **Cross-field name↔address leakage** — 0.00% in both directions. The mechanism
  is real; a true training pair exists whose only shared evidence is an S1 *name*
  token appearing in the candidate's *address*. It is simply too rare to
  accumulate gain across 10.75M rows.
* **Administrative / geographic proxies** — 0.00% for the address tail in both the
  strict and the reorder-robust form, and 0.01% for the PIN/ZIP agreement.

One caveat stated honestly: gain-based importance understates a column that is
*redundant* with a stronger one, and `addr_contain` — IDF-weighted address
containment, rank 3 in M1 at 3.63% — already sees much of what the tail columns
see. "Redundant" is better supported by this evidence than "useless". Either way
they do not pay for the per-pair set intersections they cost, and set intersection
is 38% of worker time at this scale, so **the next run should drop all four**.

What did pay off is the *phonetic* proposal, implemented as edit distance on the
romanised consonant skeleton rather than by adding a Double Metaphone dependency:
gram Jaccard is order-free and cannot separate a transposition from a near-match,
and `name_skel_lev` is M1's 7th most important feature out of 62.

---

## 4. Rejected, with the measurement

| proposal | verdict | why |
| --- | --- | --- |
| Dembczyński exact F-max under dependency | **rejected** | the whole decision-rule family spans ~0.002 here (§1.1); needs a joint `P(y)` that does not exist |
| sweep `miss_prob` | **rejected** (+0.00002) | measured over 180 configurations (§1.1) |
| tune the calibration holdout | **rejected** | ECE already 0.0003 → 0.0000; fold 4b is family-sampled from the same distribution |
| raise `prerank_max` 60 → 120 | **rejected** (bit-identical) | `max` never binds; `eps` does (§1.2) |
| char-5 sliding-window retrieval view | **rejected** | measured worse than char-4 + word tokens (0.8499 vs 0.8908 pair recall, `RESULTS.md`), and `char_wb` grams are already word-bounded, so whitespace injection *inside* a word is what they already handle |
| Double Metaphone / NYSIIS on the longest token | **replaced** | the romanised consonant skeleton already does this job for these scripts, and it is fitted from the corpus rather than tuned for English. Implemented as `name_skel_lev` + `addr_skel_jac` instead of adding a dependency that would need a licence check |
| IDF-weighted token overlap | **kept, narrowed** | IDF containment and unmatched mass already existed on the *core* name; what was missing was the **full**-token-set version and a "was anything rare shared" column, not another weighted overlap |
| raise the key-block cap 50 → 150 | **deferred** | correct instinct, but useless on its own until the per-query cut stops being arbitrary (§2.4). Both are implemented and gated behind the benchmark |

---

## 5. Not yet measured — implemented and waiting on a benchmark

Two retrieval changes are written and tested but **not** in the run above,
because both alter the raw union and so cost a full re-block (~2.5 h for the
eight partitions) before anything downstream can start. Neither is worth that
until it earns it on the standard benchmark — 4,000 held-out fold-0 India queries
against the whole 4,133,346-record S2+S3 pool, the same one every row of
`RESULTS.md` uses.

1. **Selectivity-ordered key hits + a larger block cap** (§2.4),
   `key_select=True, key_block_cap=150`.
2. **A fifth retriever: banded MinHash (LSH).** This is the one item on the list
   that attacks the actual ceiling, and the argument for it is not "MinHash
   approximates Jaccard" — the `na` view's TF-IDF cosine already does that.
   It is that **LSH retrieval is not rank-based.** All four current views are
   top-k, and in a pool where 35–44% of names repeat, a true match can sit at
   rank 200 and be invisible however good the similarity is. A banded signature
   either collides or does not, independently of how much company it has. It also
   sidesteps the rare-term query pruning, which is measured to cost 0.017 of pair
   recall on the `na` view alone (0.8809 → 0.8642 at budget 120k → 60k).

---

## 6. Runtime, and what the featurise stage taught

| stage | v2 (submitted) | v3 (`eps=0.001`) |
| --- | --- | --- |
| blocking, 8 partitions | 7,778 s | **0 s — cached and reused** |
| pre-ranker fit | ~1,200 s | **0 s — cached and reused** |
| prune, train | 227 s | 387 s |
| featurise, train | 1,412 s | _pending_ |
| featurise, test | 7,604 s | _pending_ |
| score, all partitions | 1,977 s | _pending_ |
| **total** | 15,263 s | _pending_ |

Splitting the retrieval fingerprint out (§2.3) is what makes the second column
possible: every sweep and every run on this branch reuses 5 GB of cached raw
candidates and the pre-ranker fitted to them.

### The stage was invisible, and that cost two restarts

`featurize_partition` printed **once, at the end of a partition**. A run that had
stopped making progress therefore looked exactly like a run that was busy, which
is how forty-five minutes passed on the first attempt before anything was known.
That is the same lesson the blocking stage already learned the hard way
(`PROGRESS.md` §7, fix 5) and it had not been applied here. Each shard now logs a
cumulative count and a rate, so the stage is legible in the first few minutes.

### What the cost actually is — measured, not inferred

Two of my own diagnoses were wrong, and the measurements are worth keeping
because they say where this stage's limit really is.

**Wrong #1: "the v3 features are expensive."** Timed on real India records with
the real corpus statistics:

| | µs |
| --- | --- |
| `RecordView` construction | 43.6 per record |
| `pair_feature_row` (62 columns) | 48.0 per pair |
| of which the new address skeleton + grams | 7.6 per record |
| **v3 total** | **~92 per pair** |
| v2 total, from `PROGRESS.md` §9 (41 + 57) | ~98 per pair |

The ten new columns are a **wash**. Candidate density is such that a block builds
roughly one record view per pair, so view cost sits on the critical path — and it
did not grow. No feature was trimmed on speed grounds, because none needed to be.

**Wrong #2: "the machine is out of memory."** `vm_stat` showed free pages at
0.03 GB with 7 GB in the compressor, which looks alarming and is *normal* on
macOS — the OS keeps free memory near zero by design. Summed RSS across every
process on the machine was 7.0 GB of 16. The honest signal was the one that
needed no interpretation: the parent's own resident set had been evicted to 2 MB
and it was blocked in `take_gil` for 2,324 of 2,329 samples.

**What is real.** The workers are *starved*, and how starved depends on the
worker count in a way that is not monotone:

| workers | per-worker duty cycle | throughput |
| --- | --- | --- |
| 8 (v2, smaller table) | ~15% | 13,560 pairs/s |
| 4 | ~37% | 4,850 pairs/s |
| 6, after the numpy-column change | ~61% | _pending_ |

The parent prepares every block single-threaded under the GIL, so it is the
ceiling; the changes that matter are the ones that make *it* cheaper. The
pandas-free candidate loader is the substantive one — `pd.concat` of both sources
followed by `sort_values` held two or three copies of a table that is 0.75 GB at
training scale and over 2 GB at test scale, where the rest of the module only
ever reads plain numpy arrays. Reading column by column and sorting with
`lexsort`, releasing each arrow buffer as it goes, removes those copies outright.

Capping the default worker count at 4 was a mis-step on the strength of the
duty-cycle reading, and it was reverted.

### The change that did work, and the one that invalidated its own measurement

**Workers write their own shards.** A block of ~190k pairs by 62 float32 columns
is a 47 MB result, every one of them crossed the single pipe a multiprocessing
pool shares between all its workers, and the parent had one thread to unpickle
them. On a live run the workers had computed 16M of a partition's 17.6M pairs
while the parent had written **nothing** — the compute was finished and the
results could not drain. Workers now write `X-NNNN.npy` and `K-NNNN.parquet`
themselves and return `(index, rows)`, which takes per-partition IPC from 4.4 GB
to a few hundred bytes and removes the parent-side `vstack` with it. Shards stay
in S1 order because blocks are yielded in that order and the index is the
block's position, which is what the sorted-filename read in `load_features` and
`infer.score_country` already assumed.

**And the methodological error underneath the whole episode.** Several of the
diagnostics above — the `compute_block` benchmark, the pickle benchmark, two
`iter_blocks` loads — each took a core and two to three gigabytes *while the run
they were measuring was executing*. That is enough to depress a worker duty
cycle on a ten-core machine, and it is why successive readings of the same
quantity disagreed (9%, 17%, 37%, 61%, 98%). The numbers that survived are the
ones taken on an idle machine: 160 µs/pair for `compute_block`, 0.13 s/block for
the parent, 0.06 s/block to pickle a block. The ones taken alongside the run
should not have been trusted, and the conclusions drawn from them cost three
restarts.

---

## 7. Can this reach 0.98?

**Not from this machine, and not with feature or threshold work.** The arithmetic
is unforgiving, and it is worth stating plainly rather than discovering it after
another three runs.

The reweighted score is
`0.4675·India + 0.3827·US + 0.1498·LOCO`. Even granting a *perfect* matcher —
one that predicts exactly the true matches present in the candidate list and
nothing else — the current candidate sets cap it at:

| | ceiling now (pruned) | ceiling of the raw union |
| --- | --- | --- |
| India | 0.9601 | 0.9715 |
| US | 0.9866 | 0.9909 |

A perfect matcher on the **unpruned** union scores about **0.975** reweighted, and
that is with zero errors of its own. 0.98 is *above the blocking ceiling*, so no
feature, no decision rule and no threshold can reach it. Every proposal in the
brief except the retrieval ones is bounded by that number, and most are bounded
far below it.

What the levers are actually worth, with the measurements above:

| lever | reweighted Δ | evidence |
| --- | --- | --- |
| decision layer (temperature, `miss_prob`, exact F-max under dependency) | **+0.0000** | measured, 180 configurations (§1.1) |
| pruning threshold `eps` → 0.001 | +0.002 to +0.004 | ceiling +0.0037 India (§1.2), if the matcher keeps its 96.3% capture rate |
| pruning threshold `eps` → 0.0005 | +0.004 to +0.007 | ceiling +0.0077 India |
| v3 features | +0.005 to +0.009 | judgement, against a 0.0355/0.0275 matcher gap |
| a fifth (rank-free) retriever | +0.008 to +0.013 | judgement; raw-union FCR is 0.813 on India |
| **realistic best on this laptop** | **~0.945 – 0.950** | |

The route to 0.98 is the one the pipeline was already built for and that
`PROGRESS.md` §8 documents: **the dense bi-encoder.** Its code-proving run on a
small slice took full-cluster recall from 0.9845 (sparse union) to **0.9974**
(sparse ∪ dense) — a 0.99+ ceiling, which is the only thing that makes 0.98
arithmetically possible. It needs a 24 GB GPU; on this machine's MPS the encoder
runs at 357–380 records/s, which is 17 hours for one 22M-record encode before any
retrieval happens, and the cross-encoder behind it is 17.6 days.

So: this branch banks what is available here, and the remaining 0.03 is a GPU
purchase, not an algorithm.

---

## 8. Reproducing

```bash
git checkout Experiment
export PYTHONPATH=business_entity_resolution/src
.venv/bin/python -m pytest tests -q                      # 216 + 19 tests
.venv/bin/python -u -m ber.cli all --workers 8            # ~6 h, resumable
.venv/bin/python -u -m ber.final_report
```

Probes, each reusing the cached stack and needing no re-blocking:

```bash
.venv/bin/python -u probes/probe_decision.py   # -> work/reports/probe_decision.json
.venv/bin/python -u probes/probe_prune.py      # -> work/reports/probe_prune.json
```
