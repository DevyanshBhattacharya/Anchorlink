"""Probe B — how much candidate ceiling does the pre-ranker's pruning throw away?

The first full run pruned to max=60 / min=15 / eps=0.002, which left India with
33.8 candidates per S1 and a ceiling of 0.9601 against the raw union's 0.9715.
The matcher reached 0.9246, so 0.011 of headroom is being discarded by a
threshold rather than by a model.

This scores the *cached* raw union with the *cached* pre-ranker and recomputes
the ceiling for a grid of thresholds — no retrieval, no featurising, no
training.  Ceilings are computed from labels and cluster sizes in numpy rather
than from dicts of id strings, which is what makes a 19M-row table affordable.

Writes work/reports/probe_prune.json
"""
from __future__ import annotations

import gc
import json
import sys
import time

import numpy as np

sys.path.insert(0, "business_entity_resolution/src")

from ber import config
from ber.labels import GroundTruth
from ber.model import TrainedModel
from ber.preranker import matrix, model_path, prune
from ber.retrieval import DEFAULT_BLOCKING, load_candidates
from ber.run import DEFAULT_SIZES, make_selection
from ber.splits import build_name_families

BCFG = DEFAULT_BLOCKING

#: (max, min, eps).  The first row is the configuration the last full run used.
GRID = [
    (60, 15, 0.002),
    (80, 15, 0.002),
    (120, 15, 0.002),
    (120, 15, 0.001),
    (120, 15, 0.0005),
    (120, 20, 0.0005),
    (160, 15, 0.0005),
    (160, 20, 0.0002),
    (0, 0, 0.0),        # the raw union itself, i.e. no pruning at all
]


def ceiling(inter: np.ndarray, n_true: np.ndarray, n_cand: np.ndarray) -> dict:
    """Macro F_0.5 bound and the recall numbers behind it, all in numpy.

    Per entity the bound is 1.25*|C&T| / (0.25*|T| + |C&T|): predict exactly the
    recoverable true matches and nothing else.  A singleton scores 1.0 because
    the empty prediction is available for free, and an entity with matches but no
    recoverable ones scores 0.
    """
    singleton = n_true == 0
    bound = np.zeros(len(n_true), dtype=np.float64)
    bound[singleton] = 1.0
    ns = ~singleton
    denom = 0.25 * n_true[ns] + inter[ns]
    bound[ns] = np.where(denom > 0, 1.25 * inter[ns] / np.maximum(denom, 1e-12), 0.0)
    return {
        "n": int(len(n_true)),
        "pair_recall": float(inter.sum() / max(n_true.sum(), 1)),
        "full_cluster_recall": float((inter[ns] == n_true[ns]).mean()) if ns.any() else float("nan"),
        "avg_candidates": float(n_cand.mean()),
        "f05_ceiling": float(bound.mean()),
    }


def main() -> None:
    t0 = time.time()
    fam = build_name_families("train")
    countries = sorted(fam["country"].astype(str).unique())
    sel = make_selection(fam, dict(DEFAULT_SIZES))
    gt = GroundTruth()
    model = TrainedModel.load(model_path(BCFG))
    print(f"[probe] pre-ranker: {model_path(BCFG).name}", flush=True)

    out = {"config": BCFG.retrieval_fingerprint(), "grid": [], "per_country": {}}
    for c in countries:
        ids = np.sort(sel.val[c])
        t1 = time.time()
        df = load_candidates("train", c, BCFG, pruned=False, s1_ids=ids)
        print(f"[probe] {c}: raw union {len(df):,} rows in {time.time()-t1:.0f}s",
              flush=True)
        s1 = df["s1"].to_numpy()
        y = gt.label(s1, df["src"].to_numpy(), df["cid"].to_numpy()).astype(np.int64)
        p = model.raw_predict(matrix(df))
        n_true = gt.cluster_size(ids).astype(np.int64)
        rows = []
        for mx, mn, eps in GRID:
            keep = (np.ones(len(p), dtype=bool) if mx == 0
                    else prune(df, p, mx, mn, eps))
            # per-entity recovered-true count and candidate count, aligned to ids
            inter = np.bincount(np.searchsorted(ids, s1[keep]),
                                weights=y[keep], minlength=len(ids)).astype(np.int64)
            n_cand = np.bincount(np.searchsorted(ids, s1[keep]),
                                 minlength=len(ids)).astype(np.int64)
            r = ceiling(inter, n_true, n_cand)
            r.update({"max": mx, "min": mn, "eps": eps,
                      "kept_share": float(keep.mean())})
            rows.append(r)
            print(f"  [{c}] max={mx:<4} min={mn:<3} eps={eps:<7} "
                  f"cand/S1={r['avg_candidates']:6.2f} PR={r['pair_recall']:.4f} "
                  f"FCR={r['full_cluster_recall']:.4f} ceiling={r['f05_ceiling']:.4f}",
                  flush=True)
        out["per_country"][c] = rows
        del df, y, p, s1
        gc.collect()

    out["seconds"] = round(time.time() - t0, 1)
    path = config.REPORT_DIR / "probe_prune.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwritten {path} in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
