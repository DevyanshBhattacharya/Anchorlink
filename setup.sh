#!/usr/bin/env bash
# One-time setup on a fresh clone.  Safe to re-run.
set -e
cd "$(dirname "$0")"

PY_BIN=${PY_BIN:-python3.11}
echo "== 1. Python 3.11 =="
if ! command -v "$PY_BIN" >/dev/null; then
  echo "ERROR: $PY_BIN not found."
  echo "  macOS:  brew install python@3.11"
  echo "  Ubuntu: sudo apt install python3.11 python3.11-venv"
  echo "  (3.12+ is fine too — set PY_BIN=python3.12. NOT 3.14: no LightGBM wheels.)"
  exit 1
fi
"$PY_BIN" -m venv .venv
./.venv/bin/python -m pip install -q --upgrade pip
./.venv/bin/python -m pip install -q -r business_entity_resolution/requirements.txt

echo "== 2. OpenMP (LightGBM needs it) =="
if [[ "$OSTYPE" == "darwin"* ]]; then
  brew list libomp >/dev/null 2>&1 || brew install libomp
fi

echo "== 3. the dataset =="
if [ ! -f student_resource/dataset/test/test_source1.tsv ]; then
  cat <<'NOTE'
  MISSING: student_resource/

  The challenge data is not in git (2.3 GB). Put it here so the tree looks like:

    student_resource/
      dataset/train/train_source{1,2,3}.tsv
      dataset/train/train_ground_truth.tsv
      dataset/test/test_source{1,2,3}.tsv
      utils/validate_submission.py

  Or point BER_DATA at wherever you unpacked it:
    export BER_DATA=/path/to/dataset
NOTE
  exit 1
fi
./.venv/bin/python - <<'PY'
import pathlib
base = pathlib.Path("student_resource/dataset")
want = {"train/train_source1.tsv": 210069713, "train/train_source2.tsv": 489301488,
        "train/train_source3.tsv": 503705637, "train/train_ground_truth.tsv": 127015583,
        "test/test_source1.tsv": 175022086, "test/test_source2.tsv": 509456422,
        "test/test_source3.tsv": 506002772}
bad = []
for rel, size in want.items():
    p = base / rel
    if not p.is_file():
        bad.append(f"  missing: {p}")
    elif p.stat().st_size != size:
        bad.append(f"  wrong size: {p} is {p.stat().st_size:,}, expected {size:,}")
print("  all 7 data files present and the expected size" if not bad else "\n".join(bad))
raise SystemExit(1 if bad else 0)
PY

echo "== 4. disk =="
df -h . | tail -1
echo "  the pipeline writes ~40 GB to work/ and ~1 GB to output/"

echo "== 5. tests =="
./run_tests.sh >/dev/null && echo "  all tests pass"

echo
echo "Setup complete.  Next:  ./run_all.sh"
