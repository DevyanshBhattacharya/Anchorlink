# Benchmark summary

Blocking configuration: `b3_k30_u80_df1_qb60k_key20_pr120-15-0.001`

## Blocking

| country | stage | pair_recall | full_cluster_recall | avg_candidates | reduction_ratio | f05_ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| India | raw union | 0.9282 | 0.8133 | 159.7764 | 1.0000 | 0.9715 |
| India | pre-ranked | 0.9091 | 0.7690 | 47.6131 | 1.0000 | 0.9632 |
| US | raw union | 0.9731 | 0.9163 | 159.8965 | 1.0000 | 0.9909 |
| US | pre-ranked | 0.9644 | 0.8919 | 41.9929 | 1.0000 | 0.9878 |

### Candidate set on the test split (no labels, so no recall)

| country | S1 entities | pool (S2+S3) | candidate pairs | avg_candidates | reduction_ratio | labels |
| --- | --- | --- | --- | --- | --- | --- |
| France | 259,452 | 1,434,993 | 19,940,991 | 76.86000000 | 0.99994644 | none (test-only country) |
| India | 809,986 | 4,717,565 | 39,186,058 | 48.38000000 | 0.99998974 | yes |
| US | 663,106 | 3,817,031 | 25,644,423 | 38.67000000 | 0.99998987 | yes |

## Matching (held-out fold 0, full-pool retrieval)

| country | n | macro_f05 | micro_precision | micro_recall | singleton_accuracy | avg_pred_size | rule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| India | 120,000 | 0.9313 | 0.9825 | 0.8597 | 0.9149 | 3.0310 | ('eum', 0.8, 0.4) |
| US | 120,000 | 0.9604 | 0.9878 | 0.9172 | 0.9271 | 3.2180 | ('eum', 0.8, 0.4) |

Leave-one-country-out mean: **0.88945**  
Test mix (derived from `test_source1.tsv`): {'India': 0.4675, 'US': 0.3827, 'LOCO': 0.1498}  
**Reweighted validation F_0.5: 0.9362**

### Leave-one-country-out

| fold | macro_f05 | n | rule |
| --- | --- | --- | --- |
| India->US | 0.9365 | 120,000 | ('eum', 1.0, 0.05) |
| US->India | 0.8424 | 120,000 | ('eum', 1.6, 0.4) |

## Decision layer

### India

| probabilities | best_threshold | F0.5 @ tuned threshold | F0.5 @ expected-F | expected-F temperature | F0.5 |
| --- | --- | --- | --- | --- | --- |
| m1_only | 0.6500 | 0.9262 | 0.9266 | 0.8000 | 0.9266 |
| m1_only_excl | 0.6000 | 0.9278 | 0.9283 | 0.8000 | 0.9283 |
| m2_raw | 0.6500 | 0.9298 | 0.9310 | 0.8000 | 0.9310 |
| m2_calibrated | 0.6500 | 0.9297 | 0.9310 | 0.8000 | 0.9310 |
| m2_calibrated_excl | 0.7000 | 0.9298 | 0.9310 | 0.8000 | 0.9310 |
| m2_raw_excl | 0.6500 | 0.9298 | 0.9311 | 0.8000 | 0.9311 |

### US

| probabilities | best_threshold | F0.5 @ tuned threshold | F0.5 @ expected-F | expected-F temperature | F0.5 |
| --- | --- | --- | --- | --- | --- |
| m1_only | 0.7000 | 0.9568 | 0.9570 | 0.8000 | 0.9570 |
| m1_only_excl | 0.6500 | 0.9580 | 0.9583 | 0.6000 | 0.9583 |
| m2_raw | 0.7000 | 0.9594 | 0.9607 | 1.0000 | 0.9607 |
| m2_calibrated | 0.6500 | 0.9594 | 0.9607 | 0.7000 | 0.9607 |
| m2_calibrated_excl | 0.6500 | 0.9594 | 0.9607 | 0.7000 | 0.9607 |
| m2_raw_excl | 0.7000 | 0.9594 | 0.9607 | 1.0000 | 0.9607 |

## Score by true cluster size

### India

| cluster size | n | macro_f05 | micro_precision | micro_recall |
| --- | --- | --- | --- | --- |
| singleton | 6,718 | 0.9149 | 0.0000 | 0.0000 |
| 1 match | 6,449 | 0.8379 | 0.9250 | 0.8643 |
| 2-3 | 49,191 | 0.9275 | 0.9760 | 0.8663 |
| 4-5 | 43,909 | 0.9458 | 0.9880 | 0.8597 |
| 6+ | 13,733 | 0.9503 | 0.9933 | 0.8499 |

### US

| cluster size | n | macro_f05 | micro_precision | micro_recall |
| --- | --- | --- | --- | --- |
| singleton | 6,533 | 0.9271 | 0.0000 | 0.0000 |
| 1 match | 6,545 | 0.8860 | 0.9386 | 0.9102 |
| 2-3 | 49,330 | 0.9622 | 0.9832 | 0.9250 |
| 4-5 | 43,896 | 0.9707 | 0.9925 | 0.9162 |
| 6+ | 13,696 | 0.9721 | 0.9953 | 0.9086 |

## Error analysis

### India — false positives (6,366)

| pattern | n | share |
| --- | --- | --- |
| name differs by a legal form or one token | 2,286 | 0.3591 |
| name partly overlapping | 1,273 | 0.2000 |
| no house number on one side | 1,181 | 0.1855 |
| missing address on one side | 389 | 0.0611 |
| no shared name token | 387 | 0.0608 |
| native-script name | 364 | 0.0572 |
| house numbers unrelated | 127 | 0.0199 |
| domain-style name | 124 | 0.0195 |

Examples:

* `S1-808761390` vs `S2-882575285` (p=1.0) — name partly overlapping  
  S1: **W1 North W2 Private Limited** · W3, 57-W4, W5 W6 Street, W7 Colony, W8 W9, W1, W10 W11, W1, W1 North, W12 Floor, W10 W11  
  cand: **W1 north W2 pvt. ltd.** · 57 , 57 W13, W6 STREET, W14 W9, W1 NORTH, W10 W11
* `S1-562393215` vs `S2-736324491` (p=1.0) — name differs by a legal form or one token  
  S1: **W1 W2** · No. 577/2, W3 W4 Building W5 Road, W6 Complex, W7, W8, W9 W10  
  cand: **W11  W2** · W12.NO 577/2 , W3 W4 BUILDING W5 ROAD, W6 COMPLEX, W7, W13
* `S1-343160444` vs `S2-311183051` (p=1.0) — domain-style name  
  S1: **W1 W2 Private Limited** · No.7, W3 W4 W5 Street, W6, W7, W8 W9  
  cand: **W10 W2 PRIVATE | W11.W10.W12** · NO.7, W3 W4 W5 STREET, W6, W7, W8 W9
* `S1-662373997` vs `S2-206963484` (p=1.0) — domain-style name  
  S1: **W1 W2 Private Limited** · 1711/10, W3 W4 Colony, W5 No. 4, W6 Road, W7, W8  
  cand: **W9 W2 Private | W10.W11.W12** · 1711/10, W3 W4 COLONY, W5 NO. 4, W6 ROAD, W7, W8
* `S1-960236885` vs `S3-55213029` (p=1.0) — identical name, same number  
  S1: **W1 W2 W3 Private Limited** · West W4, W5, 29/1 W6 W7 Street Flat No W8, W5, W9, W10 W11  
  cand: **W1 W2 W3 W2 Private Limited** · Door No 53 W10 W11, 29/1 W6 W7 Street Flat No W8, W5, W9, W5, W12

### India — false negatives (20,531)

| pattern | n | share |
| --- | --- | --- |
| name differs by a legal form or one token | 4,623 | 0.2252 |
| missing address on one side | 3,918 | 0.1908 |
| no shared name token | 3,182 | 0.1550 |
| no house number on one side | 2,972 | 0.1448 |
| name partly overlapping | 1,990 | 0.0969 |
| native-script name | 1,475 | 0.0718 |
| house numbers unrelated | 747 | 0.0364 |
| house numbers differ by 1-2 (same street, next door) | 532 | 0.0259 |

Examples:

* `S1-325575199` vs `S3-582705734` (p=0.0) — house numbers differ by 3-20 (same street, a few doors away)  
  S1: **W1 W2 Private Limited** · W3-15, W4 W5, W6, W7 32. W6, W7 32., W6, W7 32., W8 W9  
  cand: **W10.W11** · W3-45, W6, W12
* `S1-118198262` vs `S2-615813343` (p=0.0) — missing address on one side  
  S1: **W1 W2 Limited** · W3, West W4, W5, 4/W6 Dr W7 W8 W9  
  cand: **W1 LIMITED W10 W11** · 
* `S1-686360248` vs `S2-496914648` (p=0.0) — no shared name token  
  S1: **W1 W2 (W3) Corporation** · W4, West W5, 575, W6, W7 House 33/1, W8 W9 Road  
  cand: **W10/W9 W11** · W6, W12, 575, W4
* `S1-565481167` vs `S3-36448695` (p=0.0) — name differs by a legal form or one token  
  S1: **W1 W2 Pvt Ltd** · W3, 26, W4 Floor, W5 Building 217, W6 Street, W7, W7 W8  
  cand: **W1 Pvt Ltd [W9]** · 2-6, W10, W11
* `S1-967127284` vs `S2-725622725` (p=0.0) — house numbers differ by 1-2 (same street, next door)  
  S1: **W1 W2 Private Limited** · 4906, Behind W3 W4, W5, W6, W7  
  cand: **W8 W9 W10 W11** · 4908, W5, W7

### US — false positives (4,696)

| pattern | n | share |
| --- | --- | --- |
| name differs by a legal form or one token | 1,596 | 0.3399 |
| name partly overlapping | 700 | 0.1491 |
| missing address on one side | 677 | 0.1442 |
| house number truncated (1447 -> 447) | 432 | 0.0920 |
| no shared name token | 431 | 0.0918 |
| house numbers unrelated | 301 | 0.0641 |
| house numbers differ by 1-2 (same street, next door) | 159 | 0.0339 |
| house numbers differ by 3-20 (same street, a few doors away) | 146 | 0.0311 |

Examples:

* `S1-252735592` vs `S2-992076392` (p=1.0) — name differs by a legal form or one token  
  S1: **W1 W2 W3** · W4, 33 W5 Street, W1  
  cand: **W1 W2 W3 Inc** · 33 W5 ST, W1, W4
* `S1-197010175` vs `S2-453673792` (p=1.0) — name differs by a legal form or one token  
  S1: **W1 W2** · 16900 W3 Road, W4, W5  
  cand: **W1 W2 LLC** · 16900 W3 RD, W4, W5
* `S1-186976275` vs `S2-593118870` (p=1.0) — name differs by a legal form or one token  
  S1: **W1 W2** · 2116 W3 W4 Highway, W5, W6  
  cand: **W1 W2  (W7)** · 2116 W3 W4 HWY, W5, W6
* `S1-291062612` vs `S3-839053375` (p=1.0) — name differs by a legal form or one token  
  S1: **W1 W2 W3 of W4 W5 W3** · 151 W6 Lane, W4 W5, W7  
  cand: **W1 W2 W3 of W4 W5 W3 Corp** · 151 W6 Lane, W4 W5, W8
* `S1-39256764` vs `S3-330274012` (p=1.0) — identical name, same number  
  S1: **W1 W2 W3 W4** · 3699 W5 Avenue, W6 W6 120, W7, W8  
  cand: **W1 W2 W3 W4 W4** · 3699 W5 Avenue, # W6 120, W7, W9

### US — false negatives (19,654)

| pattern | n | share |
| --- | --- | --- |
| missing address on one side | 5,595 | 0.2847 |
| name differs by a legal form or one token | 3,435 | 0.1748 |
| no shared name token | 2,912 | 0.1482 |
| house numbers unrelated | 2,027 | 0.1031 |
| no house number on one side | 1,217 | 0.0619 |
| house numbers differ by 1-2 (same street, next door) | 1,194 | 0.0608 |
| name partly overlapping | 1,132 | 0.0576 |
| house numbers differ by 3-20 (same street, a few doors away) | 1,030 | 0.0524 |

Examples:

* `S1-591668623` vs `S3-473472141` (p=0.0) — missing address on one side  
  S1: **W1 Inc** · W2, W3, 117 W4 Road  
  cand: **W1 W5** · 
* `S1-914742846` vs `S3-963434971` (p=0.0) — no shared name token  
  S1: **W1, W2 & W3 W4 Inc** · W5, W6, 567 W6 W7 Road  
  cand: **W8.W9** · 567 W6 W7 Rd, W6, W10
* `S1-655010230` vs `S2-229684337` (p=0.0) — no house number on one side  
  S1: **W1 W2** · 245 W3 Street, W4 W5 2, W6, W7  
  cand: **W8** · W9, W3 ST, W7, W10 W11
* `S1-500137885` vs `S2-281752112` (p=0.0) — no shared name token  
  S1: **W1 W2 & W3 W4 W5 LLC** · 5250 89, W6 W6 157, W7, W8  
  cand: **W9.W10** · 5250 89, W11, W7, W8
* `S1-898452809` vs `S3-384971517` (p=0.0) — house numbers differ by 1-2 (same street, next door)  
  S1: **W1** · 35505 W2 Rd, W3, W4  
  cand: **W5 (W6: 54448)** · 35506 W2 Rd, W7/W8, W3, W9

## Ablation by phase

| phase | measured on | India | US | LOCO | reweighted | kept |
| --- | --- | --- | --- | --- | --- | --- |
| 0 — sparse blocking + LightGBM stack | full partitions, held-out fold 0 | 0.9313 | 0.9604 | 0.8894 | 0.9362 | yes — the submitted pipeline |
| 1 — dense bi-encoder | not run |  |  |  |  | no — see PROGRESS.md §8 (AWS handover) |
| 2 — cross-encoder | not run |  |  |  |  | no — see PROGRESS.md §8 (AWS handover) |

## Submission

```
{
 "countries": {
  "France": {
   "pairs_scored": 19940991,
   "ids_predicted": 839054,
   "s1": 259452
  },
  "India": {
   "pairs_scored": 39186058,
   "ids_predicted": 2441737,
   "s1": 809986
  },
  "US": {
   "pairs_scored": 25644423,
   "ids_predicted": 2144067,
   "s1": 663106
  }
 },
 "rule": "('eum', 0.8, 0.4)",
 "candidate_pairs.tsv": {
  "rows": 1732544,
  "non_empty": 1732544,
  "ids": 84771472
 },
 "matching_results.tsv": {
  "rows": 1732544,
  "non_empty": 1624146,
  "ids": 5424858
 }
}
```

### Official validator

`.venv/bin/python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids`

**PASS** (exit 0)

```
ML Challenge 2026 — submission validator
  test dir: student_resource/dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (108398 empty, 1624146 non-empty).
  candidate_pairs.tsv: 1732544 rows (0 empty, 1732544 non-empty).

PASS — no blocking issues found. Safe to submit.
```

Phase-0 end-to-end wall time: 11980 s.
