"""Probe A — the decision layer, on the cached v2 stack and v2 features.

Question: the production rule is expected-F with miss_prob = 0, i.e. it assumes
the candidate list is complete.  India's pruned full-cluster recall is 0.7537,
so for a quarter of entities it is not.  Does telling the rule that buy score?

Also extends the temperature grid downward: the full run chose T = 0.6 for both
countries, which is the edge of the searched set.

Costs nothing but one scoring pass: no re-blocking, no re-featurising, no
re-training.  Writes work/reports/probe_decision.json.
"""
from __future__ import annotations

import gc
import json
import sys
import time

import numpy as np

sys.path.insert(0, "business_entity_resolution/src")

from ber import config
from ber.decide import decide_sweep, temper
from ber.features import group_starts
from ber.io_utils import read_ground_truth
from ber.retrieval import DEFAULT_BLOCKING
from ber.run import DEFAULT_SIZES, make_selection
from ber.scoring import macro_f05_breakdown
from ber.splits import build_name_families, reweighted_score, test_mix_weights
from ber.train import Stack

BCFG = DEFAULT_BLOCKING
MODEL_DIR = config.MODEL_DIR / f"{BCFG.fingerprint()}_f2__m1120k_pre50k_m250k_cal30k"

TEMPS = (0.35, 0.45, 0.55, 0.6, 0.7, 0.8, 1.0, 1.25)
MISS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)


def main() -> None:
    t0 = time.time()
    fam = build_name_families("train")
    countries = sorted(fam["country"].astype(str).unique())
    sel = make_selection(fam, dict(DEFAULT_SIZES))
    stack = Stack.load(MODEL_DIR)
    gt_dict = read_ground_truth()
    from ber.infer import score_country

    out = {"model_dir": str(MODEL_DIR), "temps": list(TEMPS), "miss": list(MISS),
           "per_country": {}}
    for c in countries:
        K, p, pe = score_country("train", c, BCFG, stack, verbose=True)
        m = np.isin(K["s1"].to_numpy(), sel.val[c])
        K = K.loc[m].reset_index(drop=True)
        pe = np.asarray(pe[m], dtype=np.float64)
        del p
        gc.collect()
        s1 = K["s1"].to_numpy()
        ids = np.array([f"S{b}-{d}" for b, d in
                        zip(K["src"].to_numpy(), K["cid"].to_numpy())], dtype=object)
        bounds = group_starts(s1)
        keys = [f"S1-{i}" for i in sel.val[c]]
        gold = {k: set(gt_dict.get(k, ())) for k in keys}
        del K
        gc.collect()

        rows = []
        for T in TEMPS:
            t1 = time.time()
            q = temper(pe, T) if T != 1.0 else pe
            # one pass per entity covers every miss probability
            keeps = [np.zeros(len(q), dtype=bool) for _ in MISS]
            for a, b in zip(bounds[:-1], bounds[1:]):
                for j, (chosen, _) in enumerate(decide_sweep(q[a:b], MISS)):
                    if chosen:
                        keeps[j][a + np.fromiter(chosen, dtype=np.int64,
                                                 count=len(chosen))] = True
            for miss, keep in zip(MISS, keeps):
                pred: dict = {}
                for a, b in zip(s1[keep], ids[keep]):
                    pred.setdefault(f"S1-{a}", set()).add(b)
                rep = macro_f05_breakdown(pred, gold, keys)
                rows.append({"T": T, "miss": miss,
                             "macro_f05": round(rep["macro_f05"], 5),
                             "micro_p": round(rep["micro_precision"], 4),
                             "micro_r": round(rep["micro_recall"], 4),
                             "avg_pred": round(rep["avg_pred_size"], 3),
                             "singleton_acc": round(rep["singleton_accuracy"], 4)})
                print(f"  [{c}] T={T:<5} miss={miss:<5} F={rep['macro_f05']:.5f} "
                      f"P={rep['micro_precision']:.4f} R={rep['micro_recall']:.4f} "
                      f"pred={rep['avg_pred_size']:.2f}", flush=True)
            print(f"  [{c}] T={T} sweep in {time.time()-t1:.0f}s", flush=True)
            del keeps
            gc.collect()
        best = max(rows, key=lambda r: r["macro_f05"])
        base = next(r for r in rows if r["T"] == 0.6 and r["miss"] == 0.0)
        out["per_country"][c] = {"rows": rows, "best": best, "baseline_T0.6_miss0": base}
        print(f"[{c}] BEST {best}   (baseline {base})", flush=True)
        del pe, ids, gold, s1, bounds
        gc.collect()

    mix = test_mix_weights(countries)
    grid = {}
    for T in TEMPS:
        for miss in MISS:
            by = {c: next(r["macro_f05"] for r in out["per_country"][c]["rows"]
                          if r["T"] == T and r["miss"] == miss) for c in countries}
            grid[f"T={T},miss={miss}"] = round(reweighted_score(by, float("nan"), mix), 5)
    out["mix"] = mix
    out["reweighted_grid"] = grid
    out["reweighted_best"] = max(grid, key=grid.get)
    out["reweighted_best_score"] = grid[out["reweighted_best"]]
    out["reweighted_baseline"] = grid["T=0.6,miss=0.0"]
    out["seconds"] = round(time.time() - t0, 1)
    path = config.REPORT_DIR / "probe_decision.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nglobal rule: {out['reweighted_best']} -> {out['reweighted_best_score']} "
          f"(baseline T=0.6,miss=0 -> {out['reweighted_baseline']})")
    print(f"written {path} in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
