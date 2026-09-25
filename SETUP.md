# Setup — clone, run, submit

## What your machine needs

| | minimum | what we used |
| --- | --- | --- |
| Python | 3.11 or 3.12 (**not 3.14** — no LightGBM wheels) | 3.11.16 |
| RAM | 16 GB | 16 GB |
| free disk | **45 GB** (`work/` alone reaches ~40 GB) | 339 GB free |
| cores | 4+ | 10 |
| GPU | **not needed** for the scored pipeline | none used |

Phase 0 — the pipeline that produces the submission — is **CPU only**. It was trained
end to end on a 10-core laptop in about 4½ hours. A GPU is only needed for Phases 1–3,
which are optional extras (see the bottom of this file).

## 1. Get the data

The dataset is 2.3 GB and is **not** in git. Unpack it so the tree looks exactly like:

```
student_resource/
├── dataset/
│   ├── train/train_source1.tsv        210,069,713 bytes
│   ├── train/train_source2.tsv        489,301,488
│   ├── train/train_source3.tsv        503,705,637
│   ├── train/train_ground_truth.tsv   127,015,583
│   ├── test/test_source1.tsv          175,022,086
│   ├── test/test_source2.tsv          509,456,422
│   └── test/test_source3.tsv          506,002,772
└── utils/validate_submission.py
```

Keeping it somewhere else is fine — just `export BER_DATA=/path/to/dataset`.

**Never edit anything under `student_resource/`.** Nothing in the pipeline writes there.

## 2. Set up

```bash
./setup.sh
```

This creates `.venv`, installs the pinned dependencies, installs `libomp` on macOS
(LightGBM needs it), **checks all seven data files by exact byte size**, and runs the
test suite. It stops with a clear message if anything is missing.

## 3. Run

```bash
./run_all.sh                # or:  TEAM=ourteamname ./run_all.sh
```

Expect **4–5 hours** on 10 cores. You can interrupt it and run it again — every stage
caches its output under `work/`, keyed by a fingerprint of its configuration and of the
query set, so a rerun resumes rather than restarting.

Rough stage timings from our run:

| stage | time |
| --- | --- |
| parquet caches + corpus statistics + folds | 4 min |
| blocking, all 10 partitions (289M candidate pairs) | 2 h 10 min |
| pre-ranker + pruning | 15 min |
| feature computation (85M pairs, 8 workers) | 1 h 25 min |
| train M1 + M2 + calibration | 25 min |
| validation + leave-one-country-out | 45 min |
| test scoring + writing both TSVs | 25 min |

### What you get

```
output/matching_results.tsv     ← upload this to the leaderboard
output/candidate_pairs.tsv      ← the blocking set, for the package
results/BENCHMARK.md            ← the full benchmark
<team>_submission.zip           ← the complete package, structure verified
```

## 4. Check before submitting

```bash
export PYTHONPATH=business_entity_resolution/src
./.venv/bin/python -m ber.cli validate-submission --check-ids
```

Must print `PASS`. `run_all.sh` already runs this as part of the report, but run it
yourself before uploading — it is free and it is the thing that rejects submissions.

---

## What is left before we submit

The pipeline is **complete and already produces a validator-passing submission.** What
remains is short:

| # | task | who/where | time |
| --- | --- | --- | --- |
| 1 | Run `./run_all.sh` on the clone to regenerate `output/` | any 16 GB machine | 4–5 h |
| 2 | Put the real team name and members in `Documentation_template.md` | anyone | 5 min |
| 3 | `TEAM=<team_name> ./run_all.sh` so the zip is named correctly | — | — |
| 4 | Upload `output/matching_results.tsv` to the portal | — | 5 min |
| 5 | Upload `<team_name>_submission.zip` | — | 5 min |

Nothing else is required. Steps 2–5 are under half an hour of human time.

### Optional, only if there is time and a GPU

Phases 1–3 are implemented, unit-tested and benchmarked on subsamples, but **not** part
of the current score. They need a GPU because of arithmetic, not preference — we measured
the cross-encoder at **31.6 pairs/s** on this laptop, which is 17.6 days for the test set.

```bash
./aws_phases.sh          # one 24 GB GPU (A10G / L4 / 3090 class)
```

| phase | model | licence | doc's GPU estimate |
| --- | --- | --- | --- |
| 1 dense bi-encoder | BGE-M3 (568M) | MIT | 9–15 h |
| 2 cross-encoder | bge-reranker-v2-m3 (568M) | Apache-2.0 | 15–26 h |
| 3 LLM judge, uncertain band only | Qwen3-Reranker-4B | Apache-2.0 | 6–12 h |

A phase only goes into the submission **if it beats the current best on validation**, and
the leave-one-country-out fold is the one that stands in for France. `PROGRESS.md` §8 has
the full handover.

---

## If something goes wrong

| symptom | cause | fix |
| --- | --- | --- |
| `OSError: libomp.dylib` | LightGBM needs OpenMP | `brew install libomp` |
| No LightGBM/torch wheels | Python 3.14 | use 3.11 or 3.12 |
| Blocking crawls (< 500 queries/s) | two OpenMP runtimes in one process | you will see a `WARNING` from `check_openmp_health()`; run retrieval before importing LightGBM/torch — `ber.cli all` already orders itself correctly |
| Process killed during features | RAM | `--workers 4` |
| Disk full mid-run | `work/` reaches ~40 GB | free space, rerun — it resumes |
| Tests hang or segfault | PyTorch + sparse index in one pytest process | use `./run_tests.sh`, which runs two passes on purpose |
