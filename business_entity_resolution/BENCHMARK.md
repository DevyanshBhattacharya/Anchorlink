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
  S1: **Valiant Entertainment Private Limited** · Dd Trade Tower 2Nd Floor 36/2342 F-3 (Part) Kaloor Kadavanthara Ernakulam Kl, Ernakulam, Kerala  
  cand: **Valiant Entertainment Limited** · DD TRADE TOWER 2ND FLOOR 36/2342 F-3 (PART) KALOOR KADAVANTHARA ERNAKULAM KL, ERNAKULAM, Kerala
* `S1-394185437` vs `S2-514854666` (p=1.0) — name differs by a legal form or one token  
  S1: **GE Brokers Private Limited** · Pn 80 Panchaseel Nagar Nagpur, Nagpur, Maharashtra  
  cand: **GE  Brokers Limited** · PN 80 PANCHASEEL NAGAR NAGPUR, NAGPUR, Maharashtra
* `S1-32881467` vs `S2-442883886` (p=1.0) — name differs by a legal form or one token  
  S1: **Indore India Limited** · 4Th Floor, Rcm Icon, 3/1, Race Cource Road, -Opp. Abhay, Prashal Main Road, Indore, Indore, Madhya Pradesh  
  cand: **Indore India** · 4TH FLOOR, RCM ICON, 3/1, RACE COURCE ROAD, INDORE, Madhya Pradesh
* `S1-824167956` vs `S3-769162895` (p=0.9999) — name differs by a legal form or one token  
  S1: **Jain Estate Private Limited** · 23/1 Principal Khudiram Boseroad, Kolkata, Calcutta, West Bengal  
  cand: **Jain Estate Limited** · 23/1 Principal Khudiram Boseroad, Calcutta, Beadon Street, WB
* `S1-876713311` vs `S2-846129294` (p=0.9999) — name differs by a legal form or one token  
  S1: **Nath Air Center** · House No 1 Mangu Panna Landmark Near Tatesar Wala Rasta Village Jaunti Delhi, Delhi, New Delhi, Delhi  
  cand: **NATH AIR CÉNTER LLP** · HOUSE NO 1 MANGU PANNA LANDMARK NEAR TATESAR WALA RASTA VILLAGE JAUNTI DELHI, DELHI, दिल्ली

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
  S1: **Leisure Brothers Private Limited** · Tf 29 Samanvay Sequence Manjalpur Road, Vadodara, Gujarat  
  cand: **Calosynsol** · Tf 2-9 Samanvay Sequence Majnalpur Road, Baroda, Vadodara, ગુજરાત
* `S1-118198262` vs `S2-615813343` (p=0.0) — missing address on one side  
  S1: **Future Products Limited** · Calcutta, West Bengal, Kolkata, 4/1A Dr Shyama Das Row  
  cand: **FUTURE LIMITED CENTER Enterprises** · 
* `S1-711985992` vs `S2-706740051` (p=0.0) — no shared name token  
  S1: **India Maharashtra Vyapaar Private Limited** · A-19-22, 3Rd Floor, Highway Towers Pune Mumbai Road, Chinchwad, Pune, Maharashtra  
  cand: **Synsol** · Maharashtra, A-19-2, CHINCHWAD, PUNE
* `S1-393620770` vs `S3-415516197` (p=0.0) — missing address on one side  
  S1: **Ruby Pharmaceutical** · P. No. 165 2Nd Floor Rathore Nagar Vaishali Nagar, Jaipur, Rajasthan  
  cand: **Dr Ruby [Center]** · 
* `S1-254424163` vs `S2-535839838` (p=0.0) — missing address on one side  
  S1: **APS Foundation Pvt Ltd** · Flat No. 123 Sfs, Munirka Vihar, Opp. Jnu, New Delhi, South West Delhi, Delhi  
  cand: **APS APS Pvt Ltd Partners** · 

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
  S1: **Pediatric Dentistry Group of Ashland City Group** · 151 Action Lane, Ashland City, TN  
  cand: **Pediatric Dentistry Group of Ashland City Group Corp** · 151 Action Lane, Ashland City, Tennessee
* `S1-197010175` vs `S2-453673792` (p=1.0) — name differs by a legal form or one token  
  S1: **Corner Hypnosis** · 16900 Salmonberry Road, Brookings, OR  
  cand: **Corner Hypnosis LLC** · 16900 SALMONBERRY RD, BROOKINGS, OR
* `S1-252735592` vs `S2-992076392` (p=1.0) — name differs by a legal form or one token  
  S1: **Lowell Cancer Foundation** · MA, 33 Burns Street, Lowell  
  cand: **Lowell Cancer Foundation Inc** · 33 BURNS ST, LOWELL, MA
* `S1-39256764` vs `S3-330274012` (p=1.0) — identical name, same number  
  S1: **Internal Medicine Frontier Group** · 3699 Broadbridge Avenue, Unit Unit 120, Stratford, CT  
  cand: **Internal Medicine Frontier Group Group** · 3699 Broadbridge Avenue, # Unit 120, Stratford, Connecticut
* `S1-779212870` vs `S3-217534767` (p=1.0) — name differs by a legal form or one token  
  S1: **First Transit Dynamics Company** · 570 Quarry Place Court, Reisterstown, MD  
  cand: **First Transit Dynamics Co.** · 570 Quarry Place Ct, Reisterstown, Maryland

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
  S1: **Meridian Inc** · CT, Bristol, 117 Lewis Road  
  cand: **Meridian Associates** · 
* `S1-476174685` vs `S2-422130055` (p=0.0) — no house number on one side  
  S1: **Oden & Vermette Corp** · 1517 Delaware Avenue, Unit R, Tulsa, OK  
  cand: **K0RFLUXDREX** · TULSA, OK, DELAWARE AVENUE
* `S1-630430527` vs `S2-919947855` (p=0.0) — no shared name token  
  S1: **Owens & Cannon Inc.** · 1452 Ellsworth Road, Unit 1339, Mesa, AZ  
  cand: **CALOBRIXZETA** · 1452. ELLSWORTH RD, MESA, AZ
* `S1-604417590` vs `S2-621008713` (p=0.0) — missing address on one side  
  S1: **Foundry Allocation** · 1904 Terrace Court, Jeffersonville, IN  
  cand: **Foundry Center** · 
* `S1-144285388` vs `S2-836191561` (p=0.0) — missing address on one side  
  S1: **24HR Fitness** · 2326 Country Gables Drive, Phoenix, AZ  
  cand: **24HR-Center** · 

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

`/Users/devyanshbhattacharya/Vs Code/ML Challenge 26/.venv/bin/python /Users/devyanshbhattacharya/Vs Code/ML Challenge 26/student_resource/utils/validate_submission.py --matching /Users/devyanshbhattacharya/Vs Code/ML Challenge 26/output/matching_results.tsv --candidate /Users/devyanshbhattacharya/Vs Code/ML Challenge 26/output/candidate_pairs.tsv --test-dir /Users/devyanshbhattacharya/Vs Code/ML Challenge 26/student_resource/dataset/test --check-ids`

**PASS** (exit 0)

```
ML Challenge 2026 — submission validator
  test dir: /Users/devyanshbhattacharya/Vs Code/ML Challenge 26/student_resource/dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (111981 empty, 1620563 non-empty).
  candidate_pairs.tsv: 1732544 rows (0 empty, 1732544 non-empty).

PASS — no blocking issues found. Safe to submit.
```

Phase-0 end-to-end wall time: 15351 s.
