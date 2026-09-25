#!/usr/bin/env bash
# Two passes, on purpose.
#
# PyTorch, LightGBM, scikit-learn and sparse_dot_topn each ship their own OpenMP
# runtime.  Two of them in one process is pathological on macOS: the sparse top-k
# product silently drops to a single thread (a measured 25x slowdown), or the
# process deadlocks in kmp_flag_64::wait, or it segfaults.  pytest *imports every
# collected file*, so `-m torch` is not enough — pass 2 must name the file, so the
# sparse-index modules are never imported into the same process as torch.
set -e
PY=${PY:-./.venv/bin/python}
TORCH_TESTS=tests/test_dense.py
echo "== pass 1: everything except the torch tests =="
$PY -m pytest tests -q --ignore=$TORCH_TESTS "$@"
echo "== pass 2: the torch tests, in their own process =="
$PY -m pytest $TORCH_TESTS -q "$@"
echo "== both passes green =="
