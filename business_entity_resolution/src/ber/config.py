"""Paths and global knobs. Everything is overridable by environment variable."""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
def _env_path(key: str, default: Path) -> Path:
    v = os.environ.get(key)
    return Path(v).expanduser().resolve() if v else default


# src/ber/config.py -> src/ber -> src -> business_entity_resolution -> repo root
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = _env_path("BER_ROOT", PKG_DIR.parents[2])

DATA_ROOT = _env_path("BER_DATA", REPO_ROOT / "student_resource" / "dataset")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"

WORK_DIR = _env_path("BER_WORK", REPO_ROOT / "work")
OUTPUT_DIR = _env_path("BER_OUTPUT", REPO_ROOT / "output")

CACHE_DIR = WORK_DIR / "cache"
INDEX_DIR = WORK_DIR / "index"
CAND_DIR = WORK_DIR / "candidates"
MODEL_DIR = WORK_DIR / "models"
REPORT_DIR = WORK_DIR / "reports"

for _d in (WORK_DIR, OUTPUT_DIR, CACHE_DIR, INDEX_DIR, CAND_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- constants
SEED = 20260925
SOURCES = ("S2", "S3")

#: Fallback country mix for the reweighted headline number.  The real weights are
#: derived from ``test_source1.tsv`` at run time by ``splits.test_mix_weights``;
#: this constant only documents what that derivation produces on the real files.
TEST_MIX = {"India": 0.4675, "US": 0.3827, "LOCO": 0.1498}

FILES = {
    ("train", "S1"): TRAIN_DIR / "train_source1.tsv",
    ("train", "S2"): TRAIN_DIR / "train_source2.tsv",
    ("train", "S3"): TRAIN_DIR / "train_source3.tsv",
    ("test", "S1"): TEST_DIR / "test_source1.tsv",
    ("test", "S2"): TEST_DIR / "test_source2.tsv",
    ("test", "S3"): TEST_DIR / "test_source3.tsv",
}
GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"
