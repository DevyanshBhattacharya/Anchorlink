#!/usr/bin/env bash
# Phases 1-3 at full scale on a single 24 GB GPU (A10G / L4 / 3090 class).
#
# Everything below runs unchanged from what was proven on the Mac; only the
# `--scale aws` preset differs (BGE-M3 instead of a 118M encoder, 96 clusters per
# batch instead of 24, the full pool instead of a slice).
#
# Prerequisites on the box:
#   python3.11 -m venv .venv && ./.venv/bin/python -m pip install -r \
#       business_entity_resolution/requirements.txt
#   ./.venv/bin/python -m pip install torch transformers sentence-transformers faiss-cpu
#   models fetched once with the hub online, then HF_HUB_OFFLINE=1 for every run
set -e
cd "$(dirname "$0")"
export PYTHONPATH=business_entity_resolution/src
export HF_HUB_OFFLINE=1
export BER_USE_FAISS=1     # safe on Linux; on macOS faiss+torch abort the process
PY=${PY:-./.venv/bin/python}
COUNTRY=${COUNTRY:-India}

echo "== 0. Phase 0 must exist first (blocking, features, stack) =="
$PY -m ber.cli all --workers "$(nproc)"

echo "== 1. dense bi-encoder: BGE-M3, SupCon on collision batches =="
# doc estimate: encode ~22M records 3-5 h; two mining rounds 6-10 h
$PY -m ber.phase1_run --scale aws --country "$COUNTRY" --train

echo "== 2. cross-encoder: bge-reranker-v2-m3 =="
# doc estimate: fine-tune on 3-5M pairs 6-10 h; out-of-fold + test scoring 9-16 h
$PY -m ber.phase2_run --scale aws --country "$COUNTRY"

echo "== 3. the LLM judge runs only on the uncertain band =="
echo "   Qwen/Qwen3-Reranker-4B (Apache-2.0) via cross_encoder.uncertain_band()."
echo "   Keep it only if it improves the leave-one-country-out fold."

echo "== 4. rebuild the report and the package =="
./run_all.sh

cat <<'NOTE'

Reminders that are easy to get wrong on a fresh box:
  * Run every retrieval stage BEFORE anything imports LightGBM or PyTorch.
    Two OpenMP runtimes in one process make the sparse top-k product single
    threaded (25x slower, measured) and can deadlock it. `ber.cli all` already
    orders itself correctly; `ber.phase1_run` splits into two processes.
  * A phase only counts if it beats the previous best on validation, and the
    leave-one-country-out fold is the one that stands in for France.
  * Models are restricted by the allow-list in `ber/device.py`: MIT or
    Apache-2.0, at most 8B parameters. Llama, Gemma, Qwen3-8B and
    Qwen3-Reranker-8B are refused at load time.
NOTE
