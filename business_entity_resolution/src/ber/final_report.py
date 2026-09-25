"""``python -m ber.final_report`` — the benchmark summary.

Reads the cached artefacts (candidates, features, stack, outputs) and writes a
single Markdown report plus the JSON behind it: blocking quality, matching
quality with an ablation per stage, the decision-layer comparison, error
analysis, submission facts and per-stage runtime.
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from . import config
from .analyze import (decision_ablation, error_analysis, score_variants,
                      segment_report)
from .data import load_partition
from .featurize import load_features
from .io_utils import read_ground_truth
from .labels import GroundTruth
from .pipeline import decision_mask, sort_by_s1
from .report import as_markdown, blocking_report, save_json
from .retrieval import (DEFAULT_BLOCKING, cand_dir, candidates_as_dict,
                         load_candidates)
from .run import make_selection
from .scoring import macro_f05_breakdown
from .splits import build_name_families, reweighted_score, test_mix_weights
from .train import Stack


def test_blocking_stats(bcfg) -> List[Dict]:
    """Candidate-set shape on the **test** split, per country.

    There are no labels here, so recall cannot be computed — including for
    France, which exists only in test. What can be reported is what the
    organisers use `candidate_pairs.tsv` for: how many candidates each S1 gets
    and how much of the comparison space was eliminated.
    """
    import pyarrow.parquet as pq

    from .data import countries as data_countries

    rows = []
    for c in data_countries("test", "S1"):
        d = cand_dir("test", c, bcfg, pruned=True)
        parts = sorted(d.glob("S*.parquet"))
        if not parts:
            continue
        n_pairs = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
        n_s1 = len(load_partition("test", "S1", c))
        pool = len(load_partition("test", "S2", c)) + len(load_partition("test", "S3", c))
        rows.append({
            "country": c,
            "S1 entities": n_s1,
            "pool (S2+S3)": pool,
            "candidate pairs": n_pairs,
            "avg_candidates": round(n_pairs / max(n_s1, 1), 2),
            "reduction_ratio": 1.0 - n_pairs / max(n_s1 * pool, 1),
            "labels": "yes" if c != "France" else "none (test-only country)",
        })
        gc.collect()
    return rows


def _ablation(rep: Dict, phase0: Dict) -> List[Dict]:
    """One row per phase, with what it changed and where it was measured.

    Phases 1-3 are benchmarked on a slice rather than the full partitions, so
    their rows say so explicitly instead of being quoted as if they were
    comparable to the full-run numbers.
    """
    rows: List[Dict] = []
    h = rep["headline"]
    rows.append({
        "phase": "0 — sparse blocking + LightGBM stack",
        "measured on": "full partitions, held-out fold 0",
        "India": round(h["per_country"].get("India", float("nan")), 4),
        "US": round(h["per_country"].get("US", float("nan")), 4),
        "LOCO": h["loco_mean"],
        "reweighted": round(h["reweighted"], 4),
        "kept": "yes — the submitted pipeline",
    })
    for name, path, label in (
        ("1 — dense bi-encoder", "phase1_mac.json", "slice"),
        ("2 — cross-encoder", "phase2_mac_India.json", "slice"),
    ):
        f = config.REPORT_DIR / path
        if not f.exists():
            rows.append({"phase": name, "measured on": "not run",
                         "kept": "no — see PROGRESS.md §8 (AWS handover)"})
            continue
        d = json.loads(f.read_text())
        row = {"phase": name, "measured on": f"{label}: {d.get('slice', {})}"}
        if "sparse" in d:
            row["blocking pair recall (sparse)"] = round(d["sparse"]["pair_recall"], 4)
        for k in ("dense_zero_shot", "dense_tuned", "union_zero_shot", "union_tuned"):
            if k in d:
                row[k] = round(d[k]["pair_recall"], 4)
        rows.append(row)
    return rows


def _run_validator() -> Dict:
    """Run the organisers' own validator and record exactly what it said."""
    script = config.REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"
    cmd = [sys.executable, str(script),
           "--matching", str(config.OUTPUT_DIR / "matching_results.tsv"),
           "--candidate", str(config.OUTPUT_DIR / "candidate_pairs.tsv"),
           "--test-dir", str(config.TEST_DIR), "--check-ids"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return {"command": " ".join(cmd), "returncode": res.returncode,
            "passed": res.returncode == 0,
            "stdout": res.stdout.strip().splitlines()[-12:],
            "stderr": res.stderr.strip()[-500:]}


def _parse_rule(text: str | None):
    """Turn the stored rule string back into a tuple, without ``eval``."""
    if not text:
        return ("threshold", 0.5)
    import ast
    try:
        rule = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return ("threshold", 0.5)
    return tuple(rule) if isinstance(rule, (tuple, list)) else ("threshold", 0.5)


def blocking_section(bcfg, countries: Sequence[str], gt_dict, sel) -> List[Dict]:
    rows = []
    for c in countries:
        pool = len(load_partition("train", "S2", c)) + len(load_partition("train", "S3", c))
        keys = [f"S1-{i}" for i in sel.val[c]]
        for pruned, label in ((False, "raw union"), (True, "pre-ranked")):
            df = load_candidates("train", c, bcfg, columns=["s1", "src", "cid"],
                                 pruned=pruned, s1_ids=sel.val[c])
            r = blocking_report(candidates_as_dict(df), gt_dict, keys, pool)
            r["country"] = c
            r["stage"] = label
            rows.append(r)
            del df
            gc.collect()
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--validate", action="store_true", default=True,
                    help="run the official validator and record its output")
    ap.add_argument("--no-validate", dest="validate", action="store_false")
    args = ap.parse_args(argv)

    bcfg = DEFAULT_BLOCKING
    t0 = time.time()
    fam = build_name_families("train")
    countries = sorted(fam["country"].astype(str).unique())
    sel = make_selection(fam)
    gt = GroundTruth()
    gt_dict = read_ground_truth()

    phase0 = {}
    p0 = config.REPORT_DIR / "phase0_report.json"
    if p0.exists():
        phase0 = json.loads(p0.read_text())

    print("[report] blocking metrics ...", flush=True)
    rep: Dict = {"blocking": blocking_section(bcfg, countries, gt_dict, sel),
                 "matching": {}, "decision": {}, "errors": {}, "segments": {}}

    stack_dir = None
    from .features import FEATURE_VERSION
    for d in sorted((config.MODEL_DIR).glob(
            f"{bcfg.fingerprint()}_f{FEATURE_VERSION}__*")):
        if (d / "m1.pkl").exists():
            stack_dir = d
    if stack_dir is None:
        raise SystemExit("no trained stack found — run `ber.cli all` first")
    stack = Stack.load(stack_dir)
    rep["stack_dir"] = str(stack_dir)
    rep["stack"] = stack.info

    per_country = {}
    for c in countries:
        print(f"[report] {c}: loading features ...", flush=True)
        X, K = load_features("train", c, bcfg, sel.val[c])
        X, K = sort_by_s1(X, K)
        m = np.ones(len(K), dtype=bool)
        Xv, Kv = X[m], K.loc[m].reset_index(drop=True)
        del X, K
        gc.collect()
        y = gt.label(Kv["s1"].to_numpy(), Kv["src"].to_numpy(), Kv["cid"].to_numpy())
        keys = [f"S1-{i}" for i in sel.val[c]]
        print(f"[report] {c}: scoring {len(Kv):,} rows ...", flush=True)
        variants = score_variants(stack, Xv, Kv)
        print(f"[report] {c}: decision ablation ...", flush=True)
        rep["decision"][c] = decision_ablation(Kv, variants, gt_dict, keys)

        final_p = variants["m2_calibrated_excl"]
        rule = _parse_rule(phase0.get("final_rule"))
        pm = decision_mask(Kv, final_p, rule)
        pred: Dict[str, List[str]] = {k: [] for k in keys}
        sub = Kv.loc[pm]
        for a, b_, d in zip(sub["s1"].to_numpy(), sub["src"].to_numpy(),
                            sub["cid"].to_numpy()):
            pred.setdefault(f"S1-{a}", []).append(f"S{b_}-{d}")
        b = macro_f05_breakdown(pred, gt_dict, keys)
        b["rule"] = str(rule)
        per_country[c] = b
        rep["matching"][c] = b
        print(f"[report] {c}: error analysis ...", flush=True)
        rep["errors"][c] = error_analysis(Kv, final_p, y, pm, "train", c, args.examples)
        rep["segments"][c] = segment_report(Kv, final_p, y, pm, gt_dict, keys)
        del Xv, Kv, variants, final_p
        gc.collect()

    print("[report] test-split candidate stats ...", flush=True)
    mix = test_mix_weights(countries)
    loco = phase0.get("loco", {}).get("mean", {}).get("macro_f05", float("nan"))
    rep["headline"] = {
        "per_country": {c: per_country[c]["macro_f05"] for c in countries},
        "loco_mean": loco,
        "test_mix": mix,
        "reweighted": reweighted_score(
            {c: per_country[c]["macro_f05"] for c in countries}, loco, mix),
    }
    rep["loco"] = phase0.get("loco", {})
    rep["output"] = phase0.get("output", {})
    rep["ablation"] = _ablation(rep, phase0)
    try:
        rep["test_blocking"] = test_blocking_stats(bcfg)
    except Exception as exc:                      # the report must still render
        rep["test_blocking"] = [{"error": str(exc)}]
    rep["validator"] = _run_validator() if args.validate else {"skipped": True}
    rep["runtime"] = phase0.get("seconds")
    rep["seconds"] = round(time.time() - t0, 1)

    out_json = config.REPORT_DIR / "final_report.json"
    save_json(out_json, rep)
    md = render(rep, bcfg)
    out_md = Path(args.out) if args.out else config.REPORT_DIR / "final_report.md"
    out_md.write_text(md)
    print(md)
    print(f"\n[final_report] -> {out_md} and {out_json}")


def render(rep: Dict, bcfg) -> str:
    L: List[str] = ["# Benchmark summary", "",
                    f"Blocking configuration: `{bcfg.fingerprint()}`", ""]
    L += ["## Blocking", "",
          as_markdown(rep["blocking"],
                      ["country", "stage", "pair_recall", "full_cluster_recall",
                       "avg_candidates", "reduction_ratio", "f05_ceiling"]), ""]
    if rep.get("test_blocking"):
        L += ["### Candidate set on the test split (no labels, so no recall)", "",
              as_markdown(rep["test_blocking"],
                          ["country", "S1 entities", "pool (S2+S3)", "candidate pairs",
                           "avg_candidates", "reduction_ratio", "labels"],
                          floatfmt=8), ""]
    L += ["## Matching (held-out fold 0, full-pool retrieval)", "",
          as_markdown([{**v, "country": k} for k, v in rep["matching"].items()],
                      ["country", "n", "macro_f05", "micro_precision", "micro_recall",
                       "singleton_accuracy", "avg_pred_size", "rule"]), ""]
    h = rep["headline"]
    L += [f"Leave-one-country-out mean: **{h['loco_mean']}**  ",
          f"Test mix (derived from `test_source1.tsv`): "
          f"{ {k: round(v, 4) for k, v in h['test_mix'].items()} }  ",
          f"**Reweighted validation F_0.5: {h['reweighted']:.4f}**", ""]
    if rep.get("loco"):
        L += ["### Leave-one-country-out", "",
              as_markdown([{"fold": k, **v} for k, v in rep["loco"].items() if k != "mean"],
                          ["fold", "macro_f05", "n", "rule"]), ""]
    L += ["## Decision layer", ""]
    for c, rows in rep["decision"].items():
        L += [f"### {c}", "",
              as_markdown(rows, ["probabilities", "best_threshold",
                                 "F0.5 @ tuned threshold", "F0.5 @ expected-F",
                                 "expected-F temperature", "F0.5"]), ""]
    L += ["## Score by true cluster size", ""]
    for c, rows in rep["segments"].items():
        L += [f"### {c}", "",
              as_markdown(rows, ["cluster size", "n", "macro_f05",
                                 "micro_precision", "micro_recall"]), ""]
    L += ["## Error analysis", ""]
    for c, e in rep["errors"].items():
        for kind in ("false_positives", "false_negatives"):
            d = e[kind]
            L += [f"### {c} — {kind.replace('_', ' ')} ({d['n']:,})", "",
                  as_markdown(d["patterns"], ["pattern", "n", "share"]), "",
                  "Examples:", ""]
            for ex in d["examples"]:
                L += [f"* `{ex['s1']}` vs `{ex['cand']}` (p={ex['p']}) — {ex['pattern']}  ",
                      f"  S1: **{ex['s1_name']}** · {ex['s1_addr']}  ",
                      f"  cand: **{ex['cand_name']}** · {ex['cand_addr']}"]
            L += [""]
    L += ["## Ablation by phase", "",
          as_markdown(rep.get("ablation", []),
                      ["phase", "measured on", "India", "US", "LOCO", "reweighted",
                       "kept"]), ""]
    if rep.get("output"):
        L += ["## Submission", "", "```", json.dumps(rep["output"], indent=1), "```", ""]
    v = rep.get("validator") or {}
    if not v.get("skipped"):
        L += ["### Official validator", "",
              f"`{v.get('command', '')}`", "",
              f"**{'PASS' if v.get('passed') else 'FAIL'}** (exit {v.get('returncode')})",
              "", "```", "\n".join(v.get("stdout", [])), "```", ""]
    if rep.get("runtime"):
        L += [f"Phase-0 end-to-end wall time: {rep['runtime']:.0f} s.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
