#!/usr/bin/env bash
# The whole pipeline: data -> blocking -> matching -> both TSVs -> report -> zip.
#
# Resumable.  Every stage caches its output under work/ keyed by a fingerprint of
# its configuration, so if this is interrupted you just run it again and it picks
# up where it stopped.
#
#   TEAM=yourteam ./run_all.sh
set -e
cd "$(dirname "$0")"
export PYTHONPATH=business_entity_resolution/src
export HF_HUB_OFFLINE=1          # proves no network lookup happens
PY=${PY:-./.venv/bin/python}
TEAM=${TEAM:-team}
WORKERS=${WORKERS:-$(($(getconf _NPROCESSORS_ONLN) - 2))}

mkdir -p work/reports
echo "== pipeline ==  (workers=$WORKERS, expect ~4-5 h on 10 cores / 16 GB)"
$PY -u -m ber.cli all --workers "$WORKERS" 2>&1 | tee work/reports/run.log

echo "== benchmark summary =="
$PY -u -m ber.final_report 2>&1 | tee work/reports/report.log

echo "== copy the small artefacts into results/ (these are the ones git keeps) =="
mkdir -p results
cp work/reports/final_report.md   results/BENCHMARK.md
cp work/reports/final_report.json results/benchmark.json
cp work/reports/phase0_report.json results/phase0_report.json
cp work/reports/final_report.md   business_entity_resolution/BENCHMARK.md

echo "== submission zip =="
$PY -m ber.cli package --team "$TEAM"

echo
echo "Done."
echo "  leaderboard upload : output/matching_results.tsv"
echo "  full package       : ${TEAM}_submission.zip"
echo "  benchmark          : results/BENCHMARK.md"
