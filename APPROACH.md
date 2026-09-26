# How we solve this problem

## The shape of the problem

Three sources describe the same businesses with no shared identifiers. Source 1 is
deduplicated and is the query side: for each of its 1,732,544 test records we must emit
the set of Source-2/Source-3 records describing the same business — possibly empty.

Two facts about the metric decide the whole design.

**First, the score is a plain mean of independent per-entity scores.**

```
F_0.5 = 1.25 · |predicted ∩ true| / (0.25 · |true| + |predicted|)
```

averaged over *every* S1 entity, singletons included. Because the average is linear,
maximising expected F₀.₅ **separately for each entity** is exactly optimal. There is no
single global threshold that is right for every entity — the right cut-off depends on
that entity's own candidate distribution.

**Second, β = 0.5 weights precision twice as heavily as recall, but misses are not free.**
With 4 true matches: getting 3 right scores 0.94, getting all 4 plus one wrong scores
0.83, and getting only 1 of 4 scores 0.63. So we cannot simply be timid.

The naive approach — compare every S1 against every S2/S3 — is 1.73M × 10M ≈ 1.7×10¹³
comparisons. So the pipeline is **retrieve → score → decide**.

```
Normalise ──▶ Retrieve ──▶ Pre-rank ──▶ Score ──▶ Decide ──▶ two TSVs
(romanise,     (4 views,    (down to     (52       (calibrate,
 fold,          fused)       ~32/S1)      features)  exclusivity,
 skeleton)                                           expected F0.5)
```

Everything runs **per country partition**, keyed on the exact country string. France
appears only in the test set and needs no code change — it simply gets its own partition.

### Where each step lives

| step | module | what it produces |
| --- | --- | --- |
| load the TSVs safely, cache as parquet | `io_utils.py`, `data.py` | `work/cache/*.parquet` |
| corpus statistics per country | `corpus.py`, `normalize.py` | IDF, affixes, name frequencies |
| folds and splits | `splits.py` | name families, 5 disjoint roles |
| the four sparse indexes | `blocking.py` | `work/index/*.pkl` |
| key blocks | `keys.py` | capped hash → row-id blocks |
| retrieve + fuse | `retrieval.py` | raw candidate table |
| prune to the final list | `preranker.py` | `candidate_pairs.tsv` set |
| pair features | `features.py`, `featurize.py` | `work/features/X-*.npy` |
| labels | `labels.py` | y, from ground truth |
| the models | `model.py`, `train.py` | M1, M2, calibrator |
| the decision | `decide.py`, `pipeline.py` | the emitted subsets |
| scoring the test set | `infer.py` | probabilities, streamed |
| write the TSVs | `io_utils.py`, `run.py` | `output/*.tsv` |
| metrics, ablations, errors | `scoring.py`, `analyze.py`, `final_report.py` | `results/BENCHMARK.md` |
| GPU phases | `dense*.py`, `cross_encoder.py`, `phase1.py`, `phase1_run.py`, `phase2_run.py` | not in the current score |
| licence allow-list, device choice | `device.py` | refuses a disallowed model at load time |
| masking records before publishing | `redact.py`, `report.py` | `results/` carries no challenge data |

---

## 0. Reading the data without corrupting it

Two traps here cost real score, and both are one line of code.

**`NA` is a business name.** So are `NaN`, `None`, `NULL` and `inf`. pandas turns all of
them into missing values by default. Every read in this codebase goes through one keyword
set:

```python
sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, na_filter=False
```

`QUOTE_NONE` matters too: addresses contain quote characters, and the default parser eats
them.

**`csv.writer` emits CRLF.** That glues a `\r` onto the last id of every row. The
validator's prefix check still passes, so nothing complains — and the scorer silently
loses that match. Every write uses `open(..., newline="\n")` and plain string formatting.

The seven TSVs are converted once to parquet, with entity ids stored as **int32** — the
numeric part of `S2-166376419` — because five million ids cost 20 MB as int32 and about
300 MB as Python strings.

---

## 1. Normalisation — learn the language, don't hard-code it

Business names arrive as `Pvt Ltd`, `Private Limited`, `LLC Anchor Angel`,
`8913textiles.com`, `-- Northwind Peak Inc`, and a name in Devanagari. Addresses arrive as
`003182 LYNCHBURG STREET`, `1447 → 447`, `4109-4111`, `Near Fortis Hospital`.

Three layers, in increasing order of what they assume:

**Universal rules.** NFKC, romanise every script and fold accents with `anyascii`
(ISC licence — never `unidecode`, which is GPL), lowercase, strip placeholders, split
DBA/pipe aliases, reduce a domain to its label, strip leading zeros, expand number ranges,
drop immediately repeated tokens. Nothing here is language-specific.

**Corpus statistics, per country, from unlabeled records.** IDF, plus an *edge-position*
statistic: tokens that sit at the start or end of a 3+-token name in ≥85% of their
occurrences. Run on the real test files this recovers

| country | tokens found, with their share of names |
| --- | --- |
| France | sarl 21.9%, sas 14.9%, eurl 5.9%, sasu 4.3%, sci 3.8% |
| India | limited 45.1%, ltd 15.8%, llp 3.9%, sri/shri/smt, **elelpi** (romanised एलएलपी) |
| US | llc 18.7%, inc 14.4%, corp 4.5%, co 3.7% |

**with no word list and no country list.** This is the mechanism that makes France work:
we never wrote down a single French legal form.

**A consonant skeleton.** About a quarter of Indian S2/S3 names are in Devanagari,
Malayalam, Kannada or Tamil. Romanising alone is not enough — `डायनामिक` becomes
`daynamik`, which shares few character n-grams with `Dynamic`. Stripping vowels and
normalising consonant clusters collapses both to the same key:

```
"Dynamic Hospitality Private Limited"          → dnmk hsptlt prvt lmtd
"डायनामिक हॉस्पिटैलिटी प्राइवेट लिमिटेड"        → dnmk hsptlt prvt lmtd
```

---

## 2. Retrieval — four views that fail independently

One TF-IDF index per (country, source, view), over word tokens **and** word-bounded
character 4-grams. Measured on the whole India training partition (4.1M records, queries
searching the entire pool, top-20 per source):

| view | pair recall | full-cluster recall |
| --- | --- | --- |
| name + address | 0.891 | 0.737 |
| consonant skeleton | 0.829 | 0.609 |
| address only | 0.818 | 0.579 |
| name only | 0.558 | 0.295 |
| **union of the four** | **0.947** | **0.855** |

The point is not that any one is good; it is that they fail in *different places*. A
record whose name is transliterated is found by the address view; one whose address is
corrupted by the name view; a Devanagari name by the skeleton. Name-only retrieval is
weak on its own — 35–44% of S1 names are shared by another business, so a name is not an
identifier — yet it still adds 5 points of full-cluster recall to the union.

Ranks are fused with **reciprocal-rank fusion** (`1/(60 + rank)`, summed across
retrievers), which needs no score calibration between indexes that are not comparable.

**Key blocks** are unioned in on top: two exact keys per record — *(house number, name
root)* and *(sorted core-name skeleton)* — each capped at 150 records, because a key that
matches 5,000 records carries no information. They catch the case ranked retrieval misses
entirely: text corrupted past recognition (`Apex Digital` → `Apex Didt1`) while the number
and the name root survive. Keys are stored as sorted int64 hash arrays rather than a dict:
2.3M keys cost 18 MB that way and ~300 MB as a Python dict.

Two engineering details make this survive full scale:

* **Per-view document-frequency caps.** The research doc's `max_df = 2%` means ~5k
  documents on a small slice but 46k on the India pool. Worse, the skeleton view has only
  ~11k possible 3-grams, so an absolute cap tuned for the word views deleted almost its
  whole vocabulary and its pair recall collapsed 0.829 → 0.590. Caps are per retriever.
* **Rare-term query pruning.** A query keeps only its rarest terms up to a budget on
  postings visited. Cosine is dominated by rare terms anyway, and a query's L2 norm is a
  constant that cannot reorder its own top-k.

---

## 3. Pre-ranking — and what `candidate_pairs.tsv` actually is

The union is ~160 candidates per S1. A small LightGBM on the **retrieval columns alone**
(each retriever's score and rank, the RRF fusion, whether a key block fired, how many
retrievers agreed) prunes that to ~32. It costs nothing beyond what blocking already
produced.

The problem statement defines `candidate_pairs.tsv` as *"the last one: whatever your
model actually runs inference over"* — so the pre-ranker's output **is** that file.

Pruning is adaptive: keep everything above ε, never fewer than 15, never more than 60 per
entity. A hard top-k would truncate a genuinely large cluster; a pure threshold would
leave a generic-name entity with hundreds of candidates. We measured the trade-off:

| candidates per S1 | India F₀.₅ ceiling | US ceiling |
| --- | --- | --- |
| 8.4 | 0.9437 | 0.9825 |
| 16.4 | 0.9547 | 0.9876 |
| **~32 (chosen)** | **0.9601** | **0.9866** |
| 160 (no pruning) | 0.9715 | 0.9909 |

---

## 4. Scoring — 52 features, two stacked models

Every feature is a ratio, a percentile or a class index, never a raw corpus count, so it
means the same thing in a country the model never saw. **Country is never a feature** — it
is the partition key and nothing else.

Two principles shape the set:

**Never ask a model to do house-number arithmetic.** 85.9% of true pairs share a number
against 12.7% of same-name look-alikes, whose commonest pattern is a neighbouring number
on the same street. Numbers get an explicit relation class — exact after zero-stripping ·
prefix/suffix truncation (`1447 → 447`) · differ by 1–2 · differ by 3–20 · unrelated ·
missing — plus a graded log distance. A transformer reading `616` against `627` buried in
a long address will not learn this; a feature states it.

**Compare addresses by containment, not Jaccard.** One side routinely carries extra
landmarks (`Near Fortis Hospital`) or administrative units, so unmatched weight is
reported per side and the model decides what it costs.

Missing fields are never imputed — they become NaN plus a flag, and LightGBM routes NaN
natively.

### The 62 columns, by group

| # | group | what is in it |
| --- | --- | --- |
| 1–15 | **name similarity** | Jaro–Winkler, normalised Levenshtein, token-sort and token-set ratios on *core* names; character 3-gram Jaccard on the folded and skeleton views; IDF-weighted Monge–Elkan both directions with a soft abbreviation test as inner similarity; Soft TF-IDF; IDF containment and unmatched weight per side; name-frequency percentile; best similarity over DBA / pipe / domain aliases; length ratio |
| 16–20 | **name tokens** | affix relation class, unweighted token Jaccard, one-sided non-affix token counts, token-set equality |
| 21–22 | **flags** | domain-style name, native script per side |
| 23–30 | **address** | missing flags; IDF-weighted **containment** with unmatched weight per side; character 3-gram Jaccard; Monge–Elkan over alphabetic tokens with abbreviation handling; token-count ratio; locality conflict (rare alphabetic tokens on one side only) |
| 31–35 | **house numbers** | relation class, `log(1+min|Δ|)`, shared count, unmatched share per side |
| 36–37 | **cross-field** | IDF mass of one record's *name* found in the other's *address*, both directions. Measured at 0.00% of model gain — the mechanism is real but too rare to pay, and these are the first columns to drop |
| 38–39 | **name rarity** | IDF-weighted Jaccard over the **full** token sets; the largest IDF among shared non-affix tokens, normalised by the country's IDF ceiling so it transfers to an unseen country |
| 40 | **phonetics** | normalised edit distance on the romanised consonant skeleton. Gram Jaccard is order-free and cannot separate a transposition from a near-match; this is **M1's 7th most important feature of 62** |
| 41–44 | **administrative + address rarity** | address-tail Jaccard and tail-against-whole-address coverage (both 0.00% gain — redundant with IDF address containment); shared 5–6 digit ZIP/PIN by length alone; largest shared address IDF |
| 45 | **address skeleton** | 3-gram Jaccard of the address consonant skeleton, for the reason the name has one — a Devanagari address shares no token with its romanisation |
| 46–57 | **retrieval** | each retriever's score and rank, key-hit flag, RRF, how many retrievers agreed, the pre-ranker probability |
| 58–62 | **candidate list** | agreement with this S1's best candidate from the *other* source (name and address), rank and gap within the list, list size |
| +15 | **meta, added by M2** | M1's probability, rank, gap to best, score share, count of strong candidates, exclusivity-corrected probability, claim rank, competitor count, best score from the other source — plus the **shape of the list**: relative gap to the best (M2's 3rd most important of 77), step-drop to the runner-up, standard deviation, probability sum, rank within the same source, and how much probability exclusivity took away |

The list-shape columns exist because a crowded candidate list is not evidence against the
candidate at the top of it: without them M2 sees "eight close competitors" and cannot tell
a genuine ambiguity from a chain of identically named branches.

An abbreviation counts as a partial match (0.8), never equality — the test that fires on
`rd`~`road` and `bd`~`boulevard` also fires on `rd`~`reed`.

### The feature the error analysis forced

Our first full run put **34.9% of India's false positives** in one bucket. Chasing the top
example back to the raw files turned that from a label into a mechanism:

`S1-499359562` — **"«Brand» «Type» Private Limited"**, a Kochi address. Its three
true matches all keep *Private*: `"«Brand» «Type»-Private Ltd"`,
`"«Brand» «Type» Private"`, `"«Type» «Brand» Private [Limited]"`.
And `S2-83808461` — **"«Brand» «Type» Limited"**, at a byte-identical address —
matches **no S1 entity at all**.

The distractor is the only one of the four that drops *Private*. The dataset's hard
negatives are built by altering a legal-form token — and three parts of our pipeline were
blind to it at once: the core name strips it (the affix bigram is flagged),
`token_set_ratio` returns a flat 100 whenever one token set contains the other, and
*private* is common enough in Indian names that its IDF barely moves the weighted mass.

The fix is an **unweighted count of one-sided words**, on the full token set, excluding
flagged legal forms and treating `ltd`/`limited` as the same word:

| | `name_tok_only_1` | `name_tset` | `name_idf_unmatched_1` |
| --- | --- | --- | --- |
| true: «Brand» «Type»-Private Ltd | **0** | 1.00 | 0.000 |
| true: «Brand» «Type» Private | **0** | 1.00 | 0.000 |
| true: «Type» «Brand» Private [Limited] | **0** | 1.00 | 0.000 |
| **false: «Brand» «Type» Limited** | **1** | 1.00 | 0.000 |

Clean separation on one column; the two right-hand columns show how invisible it was.

### Two models, stacked out-of-fold

**M1** scores each pair from the pairwise and retrieval features. **M2** adds the nine
columns that only exist once M1 has scored the *whole list*: rank, gap to best, score
share, how many S1 entities compete for this candidate, the best score from the other
source, and the exclusivity-corrected probability.

Both use **monotone constraints** — more name similarity is never worse, more number
conflict is never better. This removes non-monotone splits that fit US/India noise and
would not transfer to France.

---

## 5. Deciding — the part most solutions get wrong

Three corrections, then an exact rule.

**Calibration.** Isotonic regression on a fold neither model trained on. Monotone, so
rankings survive.

**Exclusivity.** No S2/S3 record belongs to two S1 entities — 0 violations in all
7,638,365 training pairs. So when several S1 entities claim the same record, an
at-most-one-owner prior gives a closed-form posterior, `p' = o / (1 + Σ o)` with
`o = p/(1-p)`. If `S2-x` scores 0.90 against one S1 and 0.60 against another, they become
0.78 and 0.13. This matters most for the 35–44% of entities whose names repeat. In the
trained M2 this is the single most important feature, at 71% of total gain.

**Expected-F₀.₅ maximisation, per entity.** Under independent marginals the optimal
prediction is always a top-k prefix of the candidates sorted by probability — so only
n+1 sets need scoring, not 2ⁿ. The empty prediction competes on equal terms, so
singletons need no separate gate.

The rule behaves the way F₀.₅ should: a lone candidate at p = 0.40 gives an empty
prediction, at p = 0.55 gives a match; for `[0.90, 0.60, 0.35, 0.20]` it emits only the
first, because adding the 0.60 *lowers* expected F₀.₅.

We verified it against exhaustive search over all 2ⁿ subsets on 300 random instances, and
we rewrote it from O(n³) to a convolution — 3.1× faster, agreeing with the naive form to
7×10⁻¹⁶, with a regression test pinning both.

---

## 6. Validating — the France problem

France has **no training labels at all**. Everything about the validation design exists to
predict that transfer honestly.

**Name-family folds.** Clusters are disjoint by construction, but 35–44% of names repeat,
so two *different* businesses sharing a name would land in different folds and the model
could memorise the name. Clusters are first unioned into *name families* by their
core-name skeleton, and folds are assigned by family.

**Five disjoint roles**, because each stage's score leaks into the next if they share
data: folds 1–2 train M1, fold 3 trains the pre-ranker (its score is an M1 feature — our
first run had it in-sample and it became M1's top feature at 38.6% of gain), fold 4a
trains M2, fold 4b fits calibration, fold 0 is validation and is never seen by any fit.

**Full-pool retrieval.** Validation queries search the entire S2/S3 pool, as at test time.
Removing other folds' records would delete the look-alike businesses that make this hard.

**Leave-one-country-out.** Train on one country, score the other. This is the only honest
proxy for France, and it is not symmetric: India → US scores 0.935 while US → India scores
0.846. A model trained only on US data transfers poorly to India, which is what you would
expect given native scripts and different address structure.

**The headline number** weights each country by its share of `test_source1.tsv`, with the
share of every test-only country going to LOCO — 0.4675 India / 0.3827 US / 0.1498 LOCO,
*derived from the data* rather than written into the code.

---

## 7. Making 290 million pairs fit in 16 GB

The pipeline generates **289M candidate pairs** and computes 52 features for 85M of them.
Nothing here is exotic; it is just a set of choices that each keep one array off the heap.

| problem | what we do |
| --- | --- |
| 22M records as Python strings | store ids as int32, keep text in parquet, load one partition at a time |
| an exact name → count table costs GBs | a 2²⁴-slot uint16 hash counter per country, 33 MB; collisions only inflate a rare name slightly, which is harmless for a log-scaled rarity feature |
| IDF vocabulary has a long hapax tail | drop tokens with df < 2 — a token seen once has idf = log(N), which is *exactly* the fallback for an unseen token, so pruning is lossless and removes ~60% of the vocabulary |
| feature matrix for a country is tens of GB | shard it (`X-*.npy` + `K-*.parquet`), filter **per shard** before stacking |
| scoring needs the whole table twice | three streaming passes: M1 per shard keeping only the probability, meta over the country at once, M2 per shard again |
| the writer held ~4 GB of strings | carry candidates as `(int32, int8, int32, bool)` arrays and format ids only as each line is written |
| a stage dies half-way | every stage caches under a fingerprint of its configuration **and** of the query set, so a rerun resumes instead of restarting |

Two hot spots were rewritten after measuring, not guessing:

* **The decision rule** was O(n³) per entity. Substituting `s = a + b` turns the inner
  double sum into one convolution, and the Poisson-binomial ladders are built
  incrementally — **3.1× faster**, and exact (agreement 7×10⁻¹⁶ over 400 random instances,
  pinned by a test).
* **The candidate-text lookup** did one `np.searchsorted` per candidate. Batched per
  source, it is now one call per block.

And one finding that mattered more than all of them: **never load two OpenMP runtimes into
one process.** LightGBM, PyTorch, scikit-learn and `sparse_dot_topn` each ship their own
`libomp`. With two resident, the sparse top-k product silently falls back to a single
thread — a measured **25× slowdown**, which turned a 1.6-hour blocking pass into a
projected 16 hours — and in the worst case it deadlocks or segfaults. `ber.cli all` runs
*all* retrieval before anything touches a model, LightGBM is imported lazily, and
`check_openmp_health()` warns if the ordering is ever undone.

---

## 8. The GPU phases — built, measured, not yet in the score

All three are implemented and unit-tested. None is in the current score, and the reason is
arithmetic: the cross-encoder runs at **31.6 pairs/s** on this laptop's GPU, which is
17.6 days for the test candidate set.

**Phase 1 — dense bi-encoder** (`dense.py`, `dense_train.py`). BGE-M3 (568M, MIT), trained
with **supervised contrastive loss**, not InfoNCE: clusters here hold 3–6 records, so
InfoNCE would treat a record's own siblings as negatives. Batches are packed with clusters
the *sparse blocker already confuses*, so the hardest negatives arrive as in-batch
negatives at no extra encoding cost, and no batch mixes countries. Mined negatives scoring
above 95% of the anchor's weakest positive are dropped, which guards against label noise.
Each record is serialised with a fixed field order, an extracted `nums:` field so house
numbers are short adjacent tokens rather than digits buried in an address, and a romanised
echo for native scripts. Proven on MPS: dense pair recall **0.952 → 0.978** after training.

**Phase 2 — cross-encoder** (`cross_encoder.py`). `bge-reranker-v2-m3` (568M, Apache-2.0),
on the same XLM-R backbone, so it reads Devanagari and Malayalam — English-only
DeBERTa-v3-large is excluded for exactly that reason. It trains on *our own blocker's*
candidates, so the training distribution matches inference, plus two augmentations: copy a
positive, shift its house number by 3–20 and flip the label (the dominant hard-negative
pattern), and label-preserving noise. Its logit becomes one more column of M2.

**Phase 3 — LLM judge.** `Qwen3-Reranker-4B` (Apache-2.0) on the uncertain band only,
scored through yes/no token logits with no generation.

**The licence rule is enforced in code, not in a comment.** `device.py` holds an
allow-list and refuses anything else *at load time*, so a wrong model id fails loudly
rather than quietly reaching a submission. Explicitly refused: Llama and Gemma on licence;
Qwen3-8B (8.2B) and Qwen3-Reranker-8B (8.19B) on the 8B cap; SPLADE (CC BY-NC-SA) and
deepparse (LGPL, and pretrained on external addresses) on both. The device is picked
automatically — CUDA, then Apple MPS, then CPU — and every phase has a preset sized for
each.

Each phase is kept **only if it beats the current best on validation**, and the
leave-one-country-out fold is the one that decides, because it is the France proxy.
`./aws_phases.sh` runs the sequence at 24 GB-GPU config.

---

## Where the score actually comes from

Held-out fold 0, 120,000 entities per country, queries searching the full S2/S3 pool.

| country | macro F₀.₅ | precision | recall | candidate ceiling |
| --- | --- | --- | --- | --- |
| India | 0.9340 | 0.985 | 0.861 | 0.9649 |
| US | 0.9613 | 0.990 | 0.914 | 0.9880 |

Leave-one-country-out, the only honest proxy for France: India→US 0.9375, US→India 0.8437,
mean **0.8906**. Test-mix weighted (0.4675 India / 0.3827 US / 0.1498 LOCO, derived from
`test_source1.tsv` rather than hard-coded): **0.9380**.

How that was reached, against the previously submitted configuration:

| | India | US | LOCO mean | reweighted |
| --- | --- | --- | --- | --- |
| feature set v1 | 0.9151 | 0.9571 | 0.8904 | 0.9275 |
| feature set v2 | 0.9246 | 0.9591 | 0.8860 | 0.9320 |
| **+ 10 pair and 6 meta columns, `eps` 0.002→0.001** | 0.9321 | 0.9610 | 0.8895 | 0.9368 |
| **+ selectivity-ordered key blocks** | **0.9340** | **0.9613** | **0.8906** | **0.9380** |

Precision is near the ceiling; **recall is the binding constraint**. The split between the
two causes has moved, and it is worth stating accurately because it decides what to work
on next: India now reaches 0.9340 against a candidate ceiling of 0.9649, so the matcher
converts **96.8%** of the headroom it is given, and roughly *half* the remaining loss is
candidates that were never retrieved rather than two thirds. Further feature work is
therefore bounded — a perfect matcher on the unpruned union scores about 0.975 reweighted —
and the next real gain is in candidate generation. That is what the dense retriever in
Phase 1 is aimed at, and it is why Phase 1 is the first thing to run if a GPU appears.

The per-country figures above tune the decision rule per country. The submission applies
**one** rule to all three countries, which costs about 0.0006; see the appendix.

By cluster size:

| true matches | India F₀.₅ | US F₀.₅ |
| --- | --- | --- |
| singleton | 0.896 | 0.938 |
| 1 | 0.797 | 0.872 |
| 2–3 | 0.911 | 0.958 |
| 4–5 | 0.933 | 0.968 |
| 6+ | 0.939 | 0.969 |

Single-match entities are the hardest: there is exactly one right answer and no supporting
evidence from a sibling record.

---

## Appendix — the actual configuration

Every number below is the value in the code, not a description of it. Config id:
`b3_k30_u80_df1_qb60k_key20_kb150_ks_pr120-15-0.001`, feature set version 3.

The id has two halves on purpose. Everything before `_pr` decides the **raw union** and
everything after it decides the **pruning** that turns the union into the set the matcher
scores, so a pruning sweep reuses cached retrieval instead of repeating an hour of it per
country.

### Retrieval

| retriever | view | tokens | n-gram | df cap (abs / frac) | min df |
| --- | --- | --- | --- | --- | --- |
| `na` | folded name + address | word + char | 4 | 20,000 / 2% | 2 |
| `nm` | folded name | word + char | 4 | 20,000 / 2% | 2 |
| `ad` | folded address | word + char | 4 | 20,000 / 2% | 2 |
| `sk` | consonant skeleton of name + address | char | 3 | 200,000 / 3% | 2 |

The skeleton's cap is ten times larger on purpose: that view has only ~11k possible
3-grams, and applying the word-view cap to it collapsed its pair recall from 0.829 to
0.590.

| knob | value |
| --- | --- |
| top-k per retriever, per source | 30 |
| union cap per source | 80 |
| RRF constant *k* | 60 |
| query budget (postings visited) | 60,000 |
| max query terms | 96 |
| key blocks per query / cap per key | 20 / 150 |
| key hits kept, when a query has more than 20 | the **most selective keys'** — see below |

**Key hits are ordered by block size, not by pool index.** Hits used to come back in pool
order and the caller kept the first twenty with `np.unique`, *which sorts* — so a query
matching a 3-record key and a 140-record key kept twenty rows chosen by lowest pool index
rather than the three precise ones. Fixing that, and only then raising the cap, is worth
**+0.0152 pair recall, +0.0300 full-cluster recall and +0.0073 of candidate ceiling at an
identical candidate count** on the standard benchmark. It is also why raising the cap on
its own would have made recall *worse*: a bigger block crowds a precise hit out with its
own low-numbered rows.

### Pre-ranker

LightGBM on the **11 retrieval columns only** — each retriever's score and rank, key-hit,
RRF, retriever-agreement count. `learning_rate 0.08`, `num_leaves 63`,
`min_data_in_leaf 500`, monotone constraints on.

Pruning: keep everything with `p ≥ 0.0010`, never fewer than **15**, never more than **120**
per S1, applied per source. That lands at ~41 candidates per entity.

`eps` is the knob that binds, and `max` is not: measured on a held-out fold, 60 / 80 / 120
per S1 give **bit-identical** candidate sets, because the average list is far shorter than
the cap and the threshold decides its length. Halving `eps` from 0.002 lifts India's
candidate ceiling 0.9595 → 0.9632 for 1.6x the pairs; 0.0005 would give 0.9672 for 2.8x.

### Matcher (M1 and M2)

```
objective binary          learning_rate 0.05        num_leaves 127
min_data_in_leaf 200      feature_fraction 0.8      bagging_fraction 0.8 (freq 1)
lambda_l2 1.0             monotone_constraints_method "advanced"
early stopping 100 rounds on a 15% holdout taken by whole S1 groups
```

M1 sees 62 columns, M2 sees those 62 plus 15 meta columns. Objective is **log-loss, never
LambdaRank** — the decision layer needs probabilities, not an ordering.

### Decision

β² = 0.25. The rule search sweeps thresholds `0.30 … 0.95` step 0.025, expected-F
temperatures `{0.35, 0.45, 0.55, 0.6, 0.7, 0.8, 1.0, 1.25, 1.6}` and candidate-miss
probabilities `{0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40}`, and takes whichever scores
best on the test-mix-weighted validation. **One rule is applied to every country**,
including France, which has no labels to tune on.

`miss_prob` is the probability that blocking dropped a true match, which raises the
true-set size the rule budgets for and so makes one more predicted candidate cheaper. It
is swept because the candidate list is *not* complete — India's pruned full-cluster recall
is 0.775. On the previous, shorter candidate list the whole sweep was worth **+0.00002**
over 180 configurations; with a longer list every fold now selects a non-zero value, so it
only pays in combination. The two sweeps together cost one pass per temperature rather
than one per pair, because the expected-F computation splits into two halves that
`miss_prob` merely mixes and neither depends on it.

### Training sizes, per country

| role | fold | S1 entities |
| --- | --- | --- |
| M1 | 1, 2 | 120,000 |
| pre-ranker | 3 | 50,000 |
| M2 | 4a | ~50,000 |
| calibration | 4b | ~30,000 |
| validation | 0 | 120,000 |

Sampled by **whole name family**, so a family never straddles two roles.

---

## What we deliberately did not do

* **No transformer on the whole candidate set.** We measured the cross-encoder at
  31.6 pairs/s on this hardware — 17.6 days for the test set. It is implemented and
  benchmarked, and it runs on a GPU or not at all.
* **No connected-component clustering.** Every predicted cluster is a star around its S1,
  so a false positive cannot chain through S2–S3 links.
* **No geocoding, no address parser.** `deepparse` is LGPL and pretrained on external
  addresses; `libpostal` is trained on OpenStreetMap. Both would break the fair-play rule.
  We type each token instead and let the features compare them.
* **No country feature.** Country is the partition key. A one-hot or embedding for France
  would be undefined, since it has no training rows.
