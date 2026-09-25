# Business Entity Resolution — ML Challenge 2026

Given business records from three noisy, independent sources, find every Source-2 and
Source-3 record that refers to the same real-world business as each Source-1 record.

The pipeline is **retrieve → score → decide**, partitioned on the exact `country`
string so a country that never appears in training (France) simply gets its own
partition.

```
Normalise ──▶ Retrieve per country ──▶ Pre-rank ──▶ Match ──▶ Decide ──▶ two TSVs
(romanise,     (4 sparse views +       (last        (48 pair  (calibrate,
 fold,          key blocks, RRF)        blocking     features  exclusivity,
 skeleton)                              filter)      + LGBM)   expected F0.5)
```

---

## Reproducing the submission

```bash
# 1. environment  (Python 3.11 — LightGBM has no 3.14 wheels)
python3.11 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
# macOS only: LightGBM needs libomp
brew install libomp

# 2. data — expected at <repo>/student_resource/dataset/{train,test}
#    override with BER_DATA=/path/to/dataset

# 3. the whole pipeline, end to end (resumable: rerun to continue)
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
python -m ber.cli all

# 4. check the outputs against the official validator
python -m ber.cli validate-submission --check-ids
```

`ber.cli all` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`,
a machine-readable report to `work/reports/phase0_report.json`, and caches every
intermediate under `work/` so an interrupted run resumes where it stopped.

### Running the stages separately

```bash
python -m ber.cli prepare                       # parquet caches, corpus stats, folds
python -m ber.cli block    --split train        # candidate generation
python -m ber.cli featurize --split train       # pair features
python -m ber.cli block    --split test
python -m ber.cli featurize --split test
python -m ber.cli all                           # picks up everything already cached
```

Useful flags: `--workers N` (feature processes, default `cpu_count-2`),
`--n-m1/--n-m2/--n-cal/--n-val` (training sizes), `--no-test` (stop after validation),
`--force` (ignore caches).

### Environment variables

| variable | meaning | default |
| --- | --- | --- |
| `BER_DATA` | dataset root holding `train/` and `test/` | `<repo>/student_resource/dataset` |
| `BER_WORK` | caches, indexes, features, models, reports | `<repo>/work` |
| `BER_OUTPUT` | where the two TSVs are written | `<repo>/output` |
| `HF_HUB_OFFLINE` | must be `1`; proves no network lookup happens | — |

### Tests

```bash
./run_tests.sh          # two passes; see below for why
```

Covers the scorer (including the README's 0.714 example), the writer (LF endings, no
quoting, no spaces), normalisation, `decide()` against exhaustive search over all 2ⁿ
subsets, and a full end-to-end run on a synthetic three-country dataset whose third
country exists only in the test split — finishing with the official
`validate_submission.py`.

### One environment rule that matters

**Never load two OpenMP runtimes into one process.** PyTorch, LightGBM, scikit-learn and
`sparse_dot_topn` each ship their own `libomp`. With two of them resident on macOS the
sparse top-k product silently falls back to a single thread — a measured **25×**
slowdown, turning a 1.6-hour blocking pass into a projected 16 hours — and in the worst
case the process deadlocks inside `kmp_flag_64::wait` or segfaults.

The pipeline is built around this:

* `ber/model.py` imports LightGBM lazily, inside the functions that use it.
* `ber.cli all` runs **all** retrieval, for both splits, before anything touches a model.
* `ber.blocking.check_openmp_health()` prints a warning if a second OpenMP runtime is
  already loaded when a query starts, so the ordering cannot regress unnoticed.
* `ber.phase1_run` runs its sparse and dense stages as **separate processes**.
* `run_tests.sh` runs the PyTorch tests in their own pytest process — `-m torch` is not
  enough, because pytest imports every file it collects.

---

## How it works

### 1. Normalisation (`ber/normalize.py`)

Three layers, in increasing order of what they assume:

* **Universal rules.** NFKC, romanise every script and fold accents with `anyascii`
  (ISC licence — never `unidecode`, which is GPL), lowercase, strip placeholders
  (`<NULL>`, `##`, leading `--`/`<<`), split DBA and `|` aliases, reduce a domain to its
  label, strip leading zeros, expand number ranges, drop immediately repeated tokens.
* **Corpus statistics, per country, no labels.** IDF, and an *edge-position* statistic
  that flags tokens sitting at the start or end of 3+-token names in ≥85% of their
  occurrences. This recovers `llc/inc/corp`, `limited/ltd/llp` and `sarl/sas/eurl/sasu/sci`
  with no word list — including `elelpi`, the romanisation of एलएलपी. Flagged tokens are
  **damped to 0.1× IDF, never deleted**: a city inside a French trade name carries little
  information, but not none.
* **Views per record.** Folded text, core name (affixes damped), romanised consonant
  skeleton, house-number set, weighted address bag. "Dynamic Hospitality Private Limited"
  and its Devanagari spelling both reduce to the skeleton `dnmk hsptlt prvt lmtd`.

### 2. Candidate generation (`ber/retrieval.py`, `ber/blocking.py`, `ber/keys.py`)

Four TF-IDF retrievers per country partition, over word tokens plus word-bounded
character n-grams, each with its own document-frequency cap, fused with reciprocal-rank
fusion (k=60) and unioned with capped deterministic key blocks:

| view | what it recovers |
| --- | --- |
| name + address | the main signal; neither field alone is enough |
| name only | records whose address is corrupted or missing |
| address only | records whose name is transliterated or replaced by a trade name |
| consonant skeleton | native-script names (about a quarter of Indian S2/S3) |

Queries keep only their rarest terms, up to a budget on postings visited. The query's
L2 norm is a per-query constant and cannot reorder its own top-k, so pruning stays close
to the unpruned ranking at a fraction of the cost.

### 3. Pre-ranking (`ber/preranker.py`)

A LightGBM on the retrieval columns alone — each retriever's score and rank, the RRF
fusion, whether a key block fired, how many retrievers agreed. It costs nothing beyond
what blocking already produced and it is **the last filter before the matcher**, so its
output is exactly what `candidate_pairs.tsv` records. Pruning is adaptive: keep
everything above ε, never fewer than 10 and never more than 30 per S1 entity.

### 4. Matching (`ber/features.py`, `ber/model.py`, `ber/train.py`)

48 explicit pair features plus 9 set-level ones. Every feature is a ratio, a percentile
or a class index, never a raw corpus count, so it means the same thing in a country the
model never saw. **Country is never a feature** — it is the partition key and nothing
else.

Two rules shape the feature set:

* **Never ask a model to do house-number arithmetic.** 85.9% of true pairs share a
  number against 12.7% of same-name lookalikes, whose commonest pattern is a
  neighbouring number on the same street. Numbers get a relation class (exact,
  truncation, differ 1–2, differ 3–20, other, missing) and a graded log distance.
* **Compare addresses by containment, not Jaccard.** One side routinely carries extra
  landmarks or administrative units, so unmatched weight is reported per side.

Missing fields are never imputed: they become NaN plus a flag, and LightGBM routes NaN
natively. Monotone constraints ("more name similarity is never worse, more number
conflict is never better") remove non-monotone splits that would not transfer.

Two stages: **M1** on the pairwise and retrieval features, then **M2** on those plus the
nine columns that only exist once M1 has scored the whole list (rank, gap to best,
exclusivity-renormalised probability, competitor count, best score from the other
source). Stacking is out-of-fold by name family.

### 5. Decision (`ber/decide.py`)

The leaderboard score is a plain mean of independent per-entity scores, so maximising
expected F_0.5 separately for each S1 entity is *exactly* optimal. Under independent
marginals the optimum is always a top-k prefix, so only n+1 sets need scoring instead of
2ⁿ — verified against exhaustive search on 300 random instances.

Before that rule runs, two corrections: isotonic **calibration**, and **exclusivity** —
no S2/S3 record belongs to two S1 entities (0 violations in 7,638,365 training pairs),
so competing claims are renormalised in closed form, `p' = o / (1 + Σ o)`.

The tuned global threshold and the expected-F rule are both evaluated on the held-out
fold and the better one is used.

### 6. Validation (`ber/splits.py`)

* **Name-family GroupKFold.** Clusters are disjoint by construction, but 35–44% of S1
  names repeat, so clusters are first unioned into *name families* by their core-name
  skeleton. Without this, two different businesses sharing a name would land in
  different folds and the model could memorise the name.
* **Full-pool retrieval.** Validation queries search the entire S2/S3 pool, as at test
  time; removing other folds' records would delete the look-alike businesses that make
  the task hard.
* **Leave-one-country-out.** Train on one country, score the other. France has no
  training rows, so this cross-country gap is the only honest proxy for it.
* The headline number weights each country by its share of `test_source1.tsv`, with the
  share of every test-only country going to LOCO — 0.4675 India / 0.3827 US / 0.1498
  LOCO, derived from the data rather than written into the code.

---

## What the pipeline writes

Nothing is ever written inside the provided dataset folder. Every path comes from
`ber/config.py`:

| path | contents |
| --- | --- |
| `work/cache/` | parquet copies of the seven TSVs, corpus statistics, name families |
| `work/index/` | the sparse indexes, one file per (split, source, country, view) |
| `work/candidates/<config>/raw/` | the raw retriever union |
| `work/candidates/<config>/pruned/` | what the pre-ranker kept — this is `candidate_pairs.tsv` |
| `work/features/<config>/` | feature shards (`X-*.npy`) and their keys (`K-*.parquet`) |
| `work/models/` | the pre-ranker and the M1/M2/calibration stack |
| `work/reports/` | `phase0_report.json`, `final_report.md`, phase benchmarks |
| `output/` | `matching_results.tsv` and `candidate_pairs.tsv` |

Every stage is keyed by a configuration fingerprint and a query-set fingerprint, so a
rerun with different settings never silently reuses the wrong cache, and an interrupted
run resumes at the stage it stopped in.

## Fair play and licences

* **No external data, APIs or network lookups.** Nothing outside the provided dataset is
  read. Inference runs with `HF_HUB_OFFLINE=1`.
* **No GPL dependencies.** `anyascii` (ISC) is used for romanisation, never `unidecode`.
  All pinned libraries are MIT, BSD-3, Apache-2.0 or ISC — see `requirements.txt`.
* **No training on test records.** Labels come only from `train_ground_truth.tsv`.
  Corpus statistics computed inside a retrieval index over the records being searched
  (IDF, affixness, name frequency) are index statistics, the same thing a BM25 index
  computes, and are recomputed per split.
* **Country is an open set.** The pipeline partitions on the exact country string and
  never hard-codes, filters or one-hot-encodes a country label.
