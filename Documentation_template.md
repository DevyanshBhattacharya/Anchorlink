# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 2026-09-25

---

## 1. Executive Summary

We treat the task as **retrieve → score → decide**, partitioned on the exact `country`
string so a country that appears only in the test set (France) simply gets its own
partition with no code change. Four complementary sparse retrievers — name+address, name
only, address only, and a romanised **consonant skeleton** that bridges Devanagari and
other Indic scripts — are fused with reciprocal-rank fusion and pruned by a learned
pre-ranker, whose output *is* `candidate_pairs.tsv`. A monotone-constrained LightGBM
stack scores each pair from 48 explicit features (house numbers first-class, addresses
compared by containment), and a decision layer maximises **expected F_0.5 per entity**
after isotonic calibration and a closed-form exclusivity correction. The main
contributions are the skeleton retriever, the per-entity Bayes-optimal decision rule, and
a validation design (name-family folds + leave-one-country-out) built specifically to
predict the France transfer we cannot measure.

---

## 2. Methodology

### 2.1 Problem Analysis

Everything below was measured on the provided files.

| | train | test |
| --- | --- | --- |
| S1 records | 2,206,821 | 1,732,544 |
| S2 records | 5,034,616 | 4,887,273 |
| S3 records | 5,285,603 | 5,082,316 |
| countries | US 60%, India 40% | India 46.8%, US 38.3%, **France 15.0%** |
| S2+S3 per S1 | 4.67 | **5.75** |

* **Singletons are rare (5.58%)**, so recall on clusters carries the score; 78% of S1
  entities have 2–5 true matches. Emitting 2 of 4 correct scores only 0.83.
* **No S2/S3 record belongs to two S1 entities** — 0 violations in all 7,638,365 training
  pairs. This licenses the exclusivity correction in §5.
* **Names are not identifiers.** 35% of US and 44% of Indian S1 records share their exact
  name with another S1 record. Measured on the full India training partition, name-only
  retrieval reaches 0.558 pair recall at top-20; name+address reaches 0.891.
* **House numbers are the decision boundary.** 85.9% of true pairs share a number against
  12.7% of same-name look-alikes, whose commonest pattern is a neighbouring number on the
  same street.
* **About a quarter of Indian S2/S3 names are in native scripts.** A romanised consonant
  skeleton lifts true-pair 3-gram overlap from a median 0.19 to 0.54.
* **Postal codes are nearly absent** (0.3–1.1% of Indian addresses), so there is no
  PIN/ZIP blocking key.
* **Addresses are missing on one side** for 2.6–3.4% of S2/S3 records; S1 is always
  complete. We never impute — a missing field becomes NaN plus a flag and LightGBM routes
  NaN natively.
* **The test set is denser** (5.75 vs 4.67 records per S1), so the decision layer must not
  bake in the training cardinality prior.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned pre-ranker + gradient-boosted matcher + a
decision-theoretic output layer (hybrid).

**Core Innovation:** three things working together —

1. **A consonant-skeleton retriever** that makes native-script names reachable by a
   character index (`Dynamic Hospitality Private Limited` and its Devanagari spelling both
   become `dnmk hsptlt prvt lmtd`), unioned with three orthogonal lexical views that fail
   independently.
2. **Per-entity expected-F_0.5 maximisation.** Because the leaderboard is a plain mean of
   independent per-entity scores, maximising expected F_0.5 separately per S1 is exactly
   optimal, and under independent marginals the optimum is always a top-k prefix — so n+1
   sets need scoring, not 2ⁿ. Verified against exhaustive search on 300 random instances.
3. **Learning the language-specific parts from corpus statistics instead of lists.** An
   edge-position statistic recovers `sarl/sas/eurl/sasu/sci` for France, `limited/ltd/llp`
   for India and `llc/inc/corp` for the US with no dictionary and no country list — which
   is what makes the pipeline work on a country it never saw.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used:** four ranked retrievers per `(country, source)` partition, plus
capped deterministic key blocks. All TF-IDF over word tokens *and* word-bounded character
n-grams:

| view | text indexed | what it recovers |
| --- | --- | --- |
| `na` name+address | folded name + address, word + char-4 | the main signal |
| `nm` name only | folded name | records whose address is corrupted or missing |
| `ad` address only | folded address | records whose name is transliterated or a trade name |
| `sk` skeleton | consonant skeleton of name+address, char-3 | native-script names |

plus key blocks on *(house number, name root)* and *(sorted core-name skeleton)*, capped
at 50 records per key.

Ranks are fused with reciprocal-rank fusion (k=60). A LightGBM **pre-ranker** on the
retrieval columns alone (each retriever's score and rank, RRF, key-hit flag, how many
retrievers agreed) then prunes adaptively: keep everything above ε, never fewer than 15
and never more than 60 per S1. **That pruned set is `candidate_pairs.tsv`** — it is the
last filter before the matcher, which is how the problem statement defines the file.

**Candidate pairs generated:** see `output/candidate_pairs.tsv` and the benchmark summary
in the code package (`work/reports/final_report.md`).

**How we ensured true matches were not lost:**

* Validation queries search the **entire** S2/S3 pool, never a reduced one, so the
  distractor density matches test.
* We report pair recall, full-cluster recall and the **F_0.5 ceiling** of the candidate
  set — `1.25·|C∩T| / (0.25·|T| + |C∩T|)` averaged over entities — with every blocking
  change, because that ceiling caps the whole pipeline.
* Two knobs had to change to survive full scale: the document-frequency cap became
  absolute **and per view** (2% of a state slice is ~5k documents but 46k on the India
  pool), and queries keep only their rarest terms up to a budget on postings visited.
  Applying the word-view cap to the skeleton view — which has only ~11k possible 3-grams —
  collapsed its pair recall from 0.829 to 0.590, which is why the caps are per retriever.

---

## 4. Matching Model

**Features used** (48 pairwise + 9 set-level; every one is a ratio, a percentile or a
class index, never a raw corpus count, so it means the same thing in an unseen country):

* **Name:** Jaro–Winkler, normalised Levenshtein, token-sort and token-set ratios on
  *core* names (legal-form affixes damped, not deleted); character 3-gram Jaccard on the
  folded and skeleton views; IDF-weighted Monge–Elkan in both directions with a soft
  abbreviation test as inner similarity (`rd~road`, `pvt~private`, `bd~boulevard` — worth
  0.8, never 1.0, because it also fires on `rd~reed`); Soft TF-IDF; IDF containment and
  unmatched weight per side; name-frequency percentile within the partition; best
  similarity over DBA / `|` / domain aliases; native-script and domain flags.
* **Address:** IDF-weighted **containment** (not Jaccard — one side routinely carries
  extra landmarks or administrative units) with unmatched weight reported per side;
  character 3-gram Jaccard; Monge–Elkan over alphabetic tokens with abbreviation handling;
  a locality-conflict term for rare alphabetic tokens present on only one side; missing
  flags.
* **House numbers:** a relation class (exact after zero-stripping · prefix/suffix
  truncation · differ 1–2 · differ 3–20 · unrelated · missing), `log(1+min|Δ|)`, shared
  count, unmatched share per side. Numbers are never left to a text model.
* **Retrieval:** each retriever's score and rank, the RRF fusion, key-hit, retriever
  agreement count, and the pre-ranker probability.
* **Set-level:** agreement with the S1's best candidate from the *other* source (an S2/S3
  agreement is free evidence about both), rank and gap within the candidate list, list
  size; and after the first model has scored the list — rank, gap to best, score share,
  number of strong candidates, exclusivity-renormalised probability, how many S1 entities
  compete for this candidate, and the best score from the other source.

**Model type:** two stacked LightGBM binary classifiers with **monotone constraints**
(increasing in name/address similarity and support, decreasing in number conflict and
rank), trained out-of-fold by name family. Monotone constraints remove non-monotone splits
that fit US/India noise and would not transfer to France. Objective is log-loss, never
LambdaRank — the decision layer needs probabilities, not orderings.

**Threshold selection method:** neither a fixed threshold nor a fixed rule. On the
held-out fold we search a tuned global threshold *and* the expected-F_0.5 rule under a
temperature knob, and take whichever wins on the test-mix-weighted score. Both are
reported in the benchmark summary.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** see `work/reports/final_report.md` in the code package for the
  full table — per country, leave-one-country-out, and reweighted to the test mix
  (0.4675·India + 0.3827·US + 0.1498·LOCO, derived from `test_source1.tsv` rather than
  hard-coded).
- **Common false positives (wrong merges):** dominated by *same name, neighbouring house
  number on the same street* — the pattern the data says accounts for 29.5% of same-name
  hard negatives. Next: generic names where the address is missing on one side, and
  branches of the same chain in the same locality.
- **Common false negatives (missed matches):** native-script names whose romanisation
  diverges from the S1 spelling, domain-style names (`sunmaahospitality.com`), and heavily
  truncated addresses — the same three gaps the blocking analysis identifies, which is why
  they are the target of the dense retriever.

**Validation design.** Folds are `GroupKFold` over **name families** (clusters unioned by
their core-name skeleton), not over clusters: clusters are already disjoint, but 35–44% of
names repeat, so without family grouping two different businesses sharing a name would
land in different folds and the model could memorise the name. Validation queries search
the full pool. **Leave-one-country-out** — train on one country, score the other — is the
only honest proxy for France, and the gap between in-country and cross-country scores is
our estimate of what France costs.

---

## 6. Conclusion

Partitioning on the exact country string, learning the language-specific pieces from
corpus statistics rather than word lists, and deciding per entity with calibrated,
set-aware probabilities gives a pipeline with no country-specific code — which is what an
unseen country demands. The two lessons we would carry forward: measure the *F_0.5 ceiling*
of the candidate set with every blocking change, because it caps everything downstream; and
check that no stage's score leaks into the next, which cost us a fold and was worth it.

---

## Appendix

### A. Code Artefacts

The complete runnable pipeline ships under `code/business_entity_resolution/`:

```
src/ber/
  config.py      paths and seeds, all overridable by environment variable
  io_utils.py    safe TSV readers (dtype=str, QUOTE_NONE, keep_default_na=False)
                 and LF-only writers
  data.py        parquet cache; Partition = one (split, source, country)
  normalize.py   fold/anyascii, tokens, house_numbers, skeleton, TokenRoles
  corpus.py      per-split corpus statistics, streaming
  splits.py      name-family GroupKFold, LOCO, derived test-mix weights
  blocking.py    SparseIndex: n-gram TF-IDF, per-view df cap, rare-term query pruning
  keys.py        capped deterministic key blocks
  retrieval.py   the four retrievers + RRF + key blocks
  preranker.py   the last blocking filter and adaptive pruning
  features.py    48 pair features + 9 meta features
  featurize.py   block-parallel feature driver
  labels.py      ground truth as sorted int64 keys
  model.py       monotone LightGBM + isotonic calibration + ECE
  decide.py      expected-F_0.5 maximisation, exclusivity, temperature
  scoring.py     macro F_0.5 and the blocking ceiling
  train.py       the M1 → meta → M2 → calibration stack
  infer.py       three-pass streaming scorer
  analyze.py     ablations and error analysis
  dense.py, dense_train.py, phase1*.py   dense bi-encoder (SupCon, collision batches)
  cross_encoder.py, phase2_run.py        cross-encoder and the LLM judge band
  cli.py         python -m ber.cli {prepare, block, featurize, all, package, ...}
tests/           unit tests plus an end-to-end run on a synthetic three-country dataset
```

Reproduce both output files with:

```bash
python3.11 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
export PYTHONPATH=src HF_HUB_OFFLINE=1
python -m ber.cli all
python -m ber.cli validate-submission --check-ids
```

`README.md` in the same folder documents every stage, every environment variable and the
licence position of every dependency.

### B. Additional Results

`work/reports/final_report.md` (also copied into this package) carries the full benchmark:
blocking quality per country with reduction ratio and F_0.5 ceiling, the matching ablation
per stage, the decision-layer comparison (tuned threshold vs expected-F, with and without
calibration and exclusivity), score by true cluster size, and error analysis with worked
examples.

### C. Fair play and licences

* No external data, APIs or network lookups. Inference runs with `HF_HUB_OFFLINE=1`.
* Romanisation uses `anyascii` (ISC), **not** `unidecode` (GPL). Every pinned dependency is
  MIT, BSD-3, Apache-2.0 or ISC.
* Learned components are restricted by an allow-list in `src/ber/device.py` to MIT /
  Apache-2.0 models of at most 8B parameters; Llama, Gemma, Qwen3-8B and Qwen3-Reranker-8B
  are explicitly refused.
* No labels from the test set are used anywhere. Corpus statistics computed inside a
  retrieval index over the records being searched (IDF, affixness, name frequency) are
  index statistics, recomputed per split.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
