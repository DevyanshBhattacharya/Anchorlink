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

Ranks are fused with reciprocal-rank fusion, and capped deterministic key blocks
(house number + name root) are unioned in for records whose text is corrupted but whose
number and root survive.

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

## Where the score actually comes from

| country | macro F₀.₅ | precision | recall | singleton accuracy |
| --- | --- | --- | --- | --- |
| India | 0.9151 | 0.974 | 0.838 | 0.896 |
| US | 0.9571 | 0.988 | 0.909 | 0.938 |

Precision is already near the ceiling; **recall is the binding constraint**, and most of
what recall loses is lost in blocking, not in matching. India reaches 0.9151 against a
blocking ceiling of 0.9601 — so about a third of the remaining gap is the matcher and two
thirds is candidates that were never retrieved. That is exactly what the dense retriever
in Phase 1 is aimed at, and it is why Phase 1 is the first thing to run if a GPU appears.

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
