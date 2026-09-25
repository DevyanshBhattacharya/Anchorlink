# Benchmark summary

Blocking configuration: `b3_k30_u80_df1_qb60k_key20_pr60`

## Blocking

| country | stage | pair_recall | full_cluster_recall | avg_candidates | reduction_ratio | f05_ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| India | raw union | 0.9282 | 0.8133 | 159.7764 | 1.0000 | 0.9715 |
| India | pre-ranked | 0.9021 | 0.7537 | 33.7510 | 1.0000 | 0.9601 |
| US | raw union | 0.9731 | 0.9163 | 159.8965 | 1.0000 | 0.9909 |
| US | pre-ranked | 0.9612 | 0.8831 | 31.5998 | 1.0000 | 0.9866 |

### Candidate set on the test split (no labels, so no recall)

| country | S1 entities | pool (S2+S3) | candidate pairs | avg_candidates | reduction_ratio | labels |
| --- | --- | --- | --- | --- | --- | --- |
| France | 259,452 | 1,434,993 | 12,764,019 | 49.20000000 | 0.99996572 | none (test-only country) |
| India | 809,986 | 4,717,565 | 27,528,332 | 33.99000000 | 0.99999280 | yes |
| US | 663,106 | 3,817,031 | 20,783,504 | 31.34000000 | 0.99999179 | yes |

## Matching (held-out fold 0, full-pool retrieval)

| country | n | macro_f05 | micro_precision | micro_recall | singleton_accuracy | avg_pred_size | rule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| India | 120,000 | 0.9151 | 0.9744 | 0.8376 | 0.8961 | 2.9775 | ('eum', 0.6, 0.0) |
| US | 120,000 | 0.9571 | 0.9879 | 0.9089 | 0.9380 | 3.1888 | ('eum', 0.6, 0.0) |

Leave-one-country-out mean: **0.89043**  
Test mix (derived from `test_source1.tsv`): {'India': 0.4675, 'US': 0.3827, 'LOCO': 0.1498}  
**Reweighted validation F_0.5: 0.9275**

### Leave-one-country-out

| fold | macro_f05 | n | rule |
| --- | --- | --- | --- |
| India->US | 0.9353 | 120,000 | ('eum', 0.6, 0.0) |
| US->India | 0.8455 | 120,000 | ('eum', 1.6, 0.0) |

## Decision layer

### India

| probabilities | best_threshold | F0.5 @ tuned threshold | F0.5 @ expected-F | expected-F temperature | F0.5 |
| --- | --- | --- | --- | --- | --- |
| m1_only | 0.6000 | 0.9091 | 0.9108 | 1.0000 | 0.9108 |
| m1_only_excl | 0.6000 | 0.9106 | 0.9122 | 0.8000 | 0.9122 |
| m2_raw | 0.6000 | 0.9133 | 0.9155 | 1.0000 | 0.9155 |
| m2_calibrated | 0.6000 | 0.9131 | 0.9155 | 1.0000 | 0.9155 |
| m2_calibrated_excl | 0.6000 | 0.9132 | 0.9154 | 1.0000 | 0.9154 |
| m2_raw_excl | 0.6000 | 0.9134 | 0.9155 | 1.0000 | 0.9155 |

### US

| probabilities | best_threshold | F0.5 @ tuned threshold | F0.5 @ expected-F | expected-F temperature | F0.5 |
| --- | --- | --- | --- | --- | --- |
| m1_only | 0.6500 | 0.9536 | 0.9534 | 0.8000 | 0.9536 |
| m1_only_excl | 0.6500 | 0.9546 | 0.9546 | 0.6000 | 0.9546 |
| m2_raw | 0.6500 | 0.9561 | 0.9566 | 0.8000 | 0.9566 |
| m2_calibrated | 0.6000 | 0.9561 | 0.9567 | 0.6000 | 0.9567 |
| m2_calibrated_excl | 0.6000 | 0.9562 | 0.9568 | 0.6000 | 0.9568 |
| m2_raw_excl | 0.6500 | 0.9562 | 0.9567 | 0.6000 | 0.9567 |

## Score by true cluster size

### India

| cluster size | n | macro_f05 | micro_precision | micro_recall |
| --- | --- | --- | --- | --- |
| singleton | 6,718 | 0.8961 | 0.0000 | 0.0000 |
| 1 match | 6,449 | 0.7973 | 0.8999 | 0.8319 |
| 2-3 | 49,191 | 0.9106 | 0.9651 | 0.8450 |
| 4-5 | 43,909 | 0.9329 | 0.9817 | 0.8379 |
| 6+ | 13,733 | 0.9387 | 0.9891 | 0.8267 |

### US

| cluster size | n | macro_f05 | micro_precision | micro_recall |
| --- | --- | --- | --- | --- |
| singleton | 6,533 | 0.9380 | 0.0000 | 0.0000 |
| 1 match | 6,545 | 0.8722 | 0.9403 | 0.8947 |
| 2-3 | 49,330 | 0.9582 | 0.9831 | 0.9161 |
| 4-5 | 43,896 | 0.9677 | 0.9922 | 0.9086 |
| 6+ | 13,696 | 0.9691 | 0.9948 | 0.9002 |

## Error analysis

### India — false positives (9,139)

| pattern | n | share |
| --- | --- | --- |
| name differs by a legal form or one token | 3,185 | 0.3485 |
| no house number on one side | 1,893 | 0.2071 |
| name partly overlapping | 1,760 | 0.1926 |
| native-script name | 780 | 0.0853 |
| missing address on one side | 490 | 0.0536 |
| no shared name token | 351 | 0.0384 |
| house numbers unrelated | 226 | 0.0247 |
| house number truncated (1447 -> 447) | 150 | 0.0164 |

Examples:

* `S1-499359562` vs `S2-83808461` (p=1.0) — name differs by a legal form or one token  
  S1: **W1 W2 Private Limited** · W3 W4 Tower W5 Floor 36/2342 W6-3 (W7) W8 W9 W10 W11, W10, W12  
  cand: **W1 W2 Limited** · W3 W4 TOWER W5 FLOOR 36/2342 W6-3 (W7) W8 W9 W10 W11, W10, W12
* `S1-394185437` vs `S2-514854666` (p=1.0) — name differs by a legal form or one token  
  S1: **W1 W2 Private Limited** · W3 80 W4 Nagar W5, W5, W6  
  cand: **W1  W2 Limited** · W3 80 W4 NAGAR W5, W5, W6
* `S1-32881467` vs `S2-442883886` (p=1.0) — name differs by a legal form or one token  
  S1: **W1 W2 Limited** · W3 Floor, W4 W5, 3/1, W6 W7 Road, -Opp. W8, W9 Main Road, W1, W1, W10 W11  
  cand: **W1 W2** · W3 FLOOR, W4 W5, 3/1, W6 W7 ROAD, W1, W10 W11
* `S1-824167956` vs `S3-769162895` (p=0.9999) — name differs by a legal form or one token  
  S1: **W1 W2 Private Limited** · 23/1 W3 W4 W5, W6, W7, West W8  
  cand: **W1 W2 Limited** · 23/1 W3 W4 W5, W7, W9 Street, W10
* `S1-876713311` vs `S2-846129294` (p=0.9999) — name differs by a legal form or one token  
  S1: **W1 W2 W3** · House No 1 W4 W5 W6 Near W7 W8 W9 W10 W11 W12, W12, New W12, W12  
  cand: **W1 W2 W13 LLP** · HOUSE NO 1 W4 W5 W6 NEAR W7 W8 W9 W10 W11 W12, W12, W14

### India — false negatives (26,825)

| pattern | n | share |
| --- | --- | --- |
| name differs by a legal form or one token | 5,985 | 0.2231 |
| no shared name token | 4,283 | 0.1597 |
| missing address on one side | 4,223 | 0.1574 |
| no house number on one side | 3,840 | 0.1432 |
| native-script name | 2,847 | 0.1061 |
| name partly overlapping | 1,634 | 0.0609 |
| house numbers unrelated | 1,109 | 0.0413 |
| house numbers differ by 1-2 (same street, next door) | 833 | 0.0311 |

Examples:

* `S1-428012519` vs `S3-720474325` (p=0.0) — house numbers differ by 3-20 (same street, a few doors away)  
  S1: **W1 W2 Private Limited** · W3 29 W4 W5 W6 Road, W7, W8  
  cand: **W9** · W3 2-9 W4 W5 W10 Road, W11, W7, W12
* `S1-118198262` vs `S2-615813343` (p=0.0) — missing address on one side  
  S1: **W1 W2 Limited** · W3, West W4, W5, 4/W6 Dr W7 W8 W9  
  cand: **W1 LIMITED W10 W11** · 
* `S1-711985992` vs `S2-706740051` (p=0.0) — no shared name token  
  S1: **W1 W2 W3 Private Limited** · W4-19-22, W5 Floor, Highway W6 W7 W8 Road, W9, W7, W2  
  cand: **W10** · W2, W4-19-2, W9, W7
* `S1-393620770` vs `S3-415516197` (p=0.0) — missing address on one side  
  S1: **W1 W2** · W3. No. 165 W4 Floor W5 Nagar W6 Nagar, W7, W8  
  cand: **Dr W1 [W9]** · 
* `S1-254424163` vs `S2-535839838` (p=0.0) — missing address on one side  
  S1: **W1 W2 Pvt Ltd** · Flat No. 123 W3, W4 W5, Opp. W6, New W7, South West W7, W7  
  cand: **W1 W1 Pvt Ltd W8** · 

### US — false positives (4,633)

| pattern | n | share |
| --- | --- | --- |
| name differs by a legal form or one token | 1,512 | 0.3264 |
| name partly overlapping | 778 | 0.1679 |
| missing address on one side | 648 | 0.1399 |
| house number truncated (1447 -> 447) | 503 | 0.1086 |
| no shared name token | 354 | 0.0764 |
| house numbers unrelated | 318 | 0.0686 |
| house numbers differ by 1-2 (same street, next door) | 147 | 0.0317 |
| no house number on one side | 112 | 0.0242 |

Examples:

* `S1-291062612` vs `S3-839053375` (p=1.0) — name differs by a legal form or one token  
  S1: **W1 W2 W3 of W4 W5 W3** · 151 W6 Lane, W4 W5, W7  
  cand: **W1 W2 W3 of W4 W5 W3 Corp** · 151 W6 Lane, W4 W5, W8
* `S1-197010175` vs `S2-453673792` (p=1.0) — name differs by a legal form or one token  
  S1: **W1 W2** · 16900 W3 Road, W4, W5  
  cand: **W1 W2 LLC** · 16900 W3 RD, W4, W5
* `S1-252735592` vs `S2-992076392` (p=1.0) — name differs by a legal form or one token  
  S1: **W1 W2 W3** · W4, 33 W5 Street, W1  
  cand: **W1 W2 W3 Inc** · 33 W5 ST, W1, W4
* `S1-39256764` vs `S3-330274012` (p=1.0) — identical name, same number  
  S1: **W1 W2 W3 W4** · 3699 W5 Avenue, W6 W6 120, W7, W8  
  cand: **W1 W2 W3 W4 W4** · 3699 W5 Avenue, # W6 120, W7, W9
* `S1-779212870` vs `S3-217534767` (p=1.0) — name differs by a legal form or one token  
  S1: **W1 W2 W3 Company** · 570 W4 Place W5, W6, W7  
  cand: **W1 W2 W3 Co.** · 570 W4 Place W8, W6, W9

### US — false negatives (21,738)

| pattern | n | share |
| --- | --- | --- |
| missing address on one side | 5,789 | 0.2663 |
| no shared name token | 3,841 | 0.1767 |
| name differs by a legal form or one token | 3,601 | 0.1657 |
| house numbers unrelated | 2,241 | 0.1031 |
| no house number on one side | 1,356 | 0.0624 |
| house numbers differ by 1-2 (same street, next door) | 1,304 | 0.0600 |
| house numbers differ by 3-20 (same street, a few doors away) | 1,175 | 0.0541 |
| name partly overlapping | 985 | 0.0453 |

Examples:

* `S1-591668623` vs `S3-473472141` (p=0.0) — missing address on one side  
  S1: **W1 Inc** · W2, W3, 117 W4 Road  
  cand: **W1 W5** · 
* `S1-476174685` vs `S2-422130055` (p=0.0) — no house number on one side  
  S1: **W1 & W2 Corp** · 1517 W3 Avenue, W4 W5, W6, W7  
  cand: **W8** · W6, W7, W3 AVENUE
* `S1-630430527` vs `S2-919947855` (p=0.0) — no shared name token  
  S1: **W1 & W2 Inc.** · 1452 W3 Road, W4 1339, W5, W6  
  cand: **W7** · 1452. W3 RD, W5, W6
* `S1-604417590` vs `S2-621008713` (p=0.0) — missing address on one side  
  S1: **W1 W2** · 1904 W3 W4, W5, W6  
  cand: **W1 W7** · 
* `S1-144285388` vs `S2-836191561` (p=0.0) — missing address on one side  
  S1: **W1 W2** · 2326 W3 W4 Drive, W5, W6  
  cand: **W1-W7** · 

## Ablation by phase

| phase | measured on | India | US | LOCO | reweighted | kept |
| --- | --- | --- | --- | --- | --- | --- |
| 0 — sparse blocking + LightGBM stack | full partitions, held-out fold 0 | 0.9151 | 0.9571 | 0.8904 | 0.9275 | yes — the submitted pipeline |
| 1 — dense bi-encoder | not run |  |  |  |  | no — see PROGRESS.md §8 (AWS handover) |
| 2 — cross-encoder | not run |  |  |  |  | no — see PROGRESS.md §8 (AWS handover) |

## Submission

```
{
 "countries": {
  "France": {
   "pairs_scored": 12764019,
   "ids_predicted": 822625,
   "s1": 259452
  },
  "India": {
   "pairs_scored": 27528332,
   "ids_predicted": 2414102,
   "s1": 809986
  },
  "US": {
   "pairs_scored": 20783504,
   "ids_predicted": 2122575,
   "s1": 663106
  }
 },
 "rule": "('eum', 0.6, 0.0)",
 "candidate_pairs.tsv": {
  "rows": 1732544,
  "non_empty": 1732544,
  "ids": 61075855
 },
 "matching_results.tsv": {
  "rows": 1732544,
  "non_empty": 1620563,
  "ids": 5359302
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
  matching_results.tsv: 1732544 rows (111981 empty, 1620563 non-empty).
  candidate_pairs.tsv: 1732544 rows (0 empty, 1732544 non-empty).

PASS — no blocking issues found. Safe to submit.
```

Phase-0 end-to-end wall time: 15351 s.
