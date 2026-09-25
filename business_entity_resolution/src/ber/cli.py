"""Command line entry points.  Run with ``python -m ber.cli <command>``.

Every command is resumable: it skips work whose output already exists unless
``--force`` is given.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from . import config
from .corpus import build_roles, describe_affixes, roles_path
from .data import build_all_caches, countries as data_countries
from .retrieval import BlockingConfig, DEFAULT_BLOCKING


def _countries(split: str, arg: str | None) -> List[str]:
    if arg:
        return [c for c in arg.split(",") if c]
    return data_countries(split, "S1")


def cmd_prepare(args) -> None:
    t0 = time.time()
    sizes = build_all_caches(force=args.force)
    print(f"[prepare] parquet caches: "
          f"{sum(sizes.values())/1e6:.0f} MB in {time.time()-t0:.0f}s")
    roles = build_roles("train", force=args.force)
    build_roles("test", force=args.force)
    from .splits import build_name_families
    fam = build_name_families("train", roles=roles, force=args.force)
    print(f"[prepare] name families: {fam['family'].nunique():,} for {len(fam):,} S1 records")
    for c, v in sorted(describe_affixes(roles, 12).items()):
        print(f"[prepare] affixes {c}: {', '.join(v)}")
    print(f"[prepare] done in {time.time()-t0:.0f}s")


def cmd_block(args) -> None:
    from .retrieval import retrieve_partition
    from .splits import build_name_families, fold_ids, sample_ids
    roles = build_roles(args.split)
    bcfg = DEFAULT_BLOCKING
    q_by_country = None
    if args.sample:
        fam = build_name_families("train")
        q_by_country = {}
        for c in _countries(args.split, args.countries):
            ids = np.concatenate([fold_ids(fam, f, country=c) for f in range(5)])
            q_by_country[c] = sample_ids(ids, args.sample)
    for c in _countries(args.split, args.countries):
        retrieve_partition(args.split, c, bcfg, roles,
                           q_ids=None if q_by_country is None else q_by_country[c],
                           force=args.force)
        gc.collect()


def cmd_featurize(args) -> None:
    from .featurize import featurize_partition
    bcfg = DEFAULT_BLOCKING
    rp = roles_path(args.split)
    for c in _countries(args.split, args.countries):
        featurize_partition(args.split, c, bcfg, rp, block_s1=args.block,
                            n_workers=args.workers, force=args.force)
        gc.collect()


def cmd_all(args) -> None:
    """The whole phase-0 pipeline, end to end.  Resumable at every stage."""
    import numpy as np

    from .labels import GroundTruth
    from .io_utils import read_ground_truth
    from .report import as_markdown, save_json
    from .run import (DEFAULT_SIZES, blocking_metrics, make_selection, run_loco,
                      selection_queries, stage_block, stage_featurize,
                      stage_fit_preranker, stage_prune, validate, write_outputs)
    from .splits import build_name_families, reweighted_score, test_mix_weights
    from .train import Stack, fit_stack

    t0 = time.time()
    bcfg = DEFAULT_BLOCKING
    sizes = {**DEFAULT_SIZES}
    for k in ("n_m1", "n_pre", "n_m2", "n_cal", "n_val"):
        v = getattr(args, k, None)
        if v is not None:
            sizes[k] = v
    print(f"[all] blocking config: {bcfg.fingerprint()}")
    print(f"[all] sizes: {sizes}")

    cmd_prepare(argparse.Namespace(force=False))
    fam = build_name_families("train")
    train_countries = sorted(fam["country"].astype(str).unique())
    test_countries = _countries("test", args.countries)
    sel = make_selection(fam, sizes)
    for c in train_countries:
        print(f"[all] {c}: m1={len(sel.m1[c]):,} pre={len(sel.pre[c]):,} "
              f"m2={len(sel.m2[c]):,} cal={len(sel.cal[c]):,} val={len(sel.val[c]):,}")

    # ---- all retrieval first -------------------------------------------
    # Blocking needs no model, and doing every partition before anything imports
    # LightGBM keeps a second OpenMP runtime out of this process.  With two
    # copies of libomp loaded, the sparse top-k product falls back to a single
    # thread and blocking runs about ten times slower (see PROGRESS.md §7).
    stage_block("train", bcfg, train_countries, selection_queries(sel, train_countries))
    if not args.no_test:
        stage_block("test", bcfg, test_countries, None)

    gt = GroundTruth()
    gt_dict = read_ground_truth()
    pre = stage_fit_preranker(bcfg, train_countries, sel, gt)
    stage_prune("train", bcfg, train_countries, pre)
    stage_featurize("train", bcfg, train_countries, n_workers=args.workers)

    report = {"blocking": {"raw": blocking_metrics("train", bcfg, train_countries,
                                                   gt_dict, sel, pruned=False),
                           "pruned": blocking_metrics("train", bcfg, train_countries,
                                                      gt_dict, sel, pruned=True)}}
    print(as_markdown(report["blocking"]["pruned"],
                      ["country", "stage", "pair_recall", "full_cluster_recall",
                       "avg_candidates", "reduction_ratio", "f05_ceiling"]))

    # the stack depends on how much data it saw, so the cache key carries the sizes
    size_tag = "_".join(f"{k[2:]}{sizes[k] // 1000}k" if sizes[k] >= 1000
                        else f"{k[2:]}{sizes[k]}"
                        for k in ("n_m1", "n_pre", "n_m2", "n_cal"))
    from .features import FEATURE_VERSION
    model_dir = (config.MODEL_DIR /
                 f"{bcfg.fingerprint()}_f{FEATURE_VERSION}__{size_tag}")
    if (model_dir / "m1.pkl").exists() and not args.force:
        stack = Stack.load(model_dir)
        print("[all] loaded cached stack")
    else:
        stack = fit_stack("train", train_countries, bcfg, sel, gt)
        stack.save(model_dir)
    report["stack"] = stack.info

    val = validate(bcfg, stack, train_countries, sel, gt_dict)
    report["validation"] = {"per_country": val["per_country"],
                            "decision": val["decision"]}
    loco = run_loco(bcfg, train_countries, sel, gt, gt_dict) if args.loco else {}
    report["loco"] = loco

    mix = test_mix_weights(train_countries)
    by_country = {c: val["per_country"][c]["macro_f05"] for c in train_countries}
    loco_mean = loco.get("mean", {}).get("macro_f05", float("nan"))
    report["headline"] = {
        "per_country": by_country,
        "loco_mean": loco_mean,
        "test_mix": mix,
        "reweighted": reweighted_score(by_country, loco_mean, mix),
    }
    print(f"[all] test mix (derived from test_source1.tsv): "
          f"{ {k: round(v, 3) for k, v in mix.items()} }")
    print(f"[all] reweighted validation F0.5 = {report['headline']['reweighted']:.4f}")

    rules = {c: val["per_country"][c]["rule"] for c in train_countries}
    print(f"[all] per-country decision rules: {rules}")
    # one rule for the whole test set: the one with the best reweighted score
    rule = _pick_rule(val, train_countries, gt_dict, sel, mix)
    report["final_rule"] = str(rule)
    print(f"[all] final decision rule: {rule}")

    if not args.no_test:
        stage_prune("test", bcfg, test_countries, pre)
        stage_featurize("test", bcfg, test_countries, n_workers=args.workers)
        report["output"] = write_outputs(bcfg, stack, rule, test_countries)

    report["seconds"] = round(time.time() - t0, 1)
    save_json(config.REPORT_DIR / "phase0_report.json", report)
    print(f"[all] done in {report['seconds']:.0f}s -> "
          f"{config.REPORT_DIR / 'phase0_report.json'}")


def _pick_rule(val, countries, gt_dict, sel, mix=None):
    """The single decision rule that maximises the test-mix-weighted score."""
    from .splits import reweighted_score
    tables = val["decision"]
    common = set.intersection(*(set(tables[c]) for c in countries)) if countries else set()
    best, best_score = None, -1.0
    for key in sorted(common):
        by_country = {c: tables[c][key] for c in countries}
        s = reweighted_score(by_country, float("nan"), mix)
        if s > best_score:
            best, best_score = key, s
    if not best:
        return ("threshold", 0.5)
    import ast
    return tuple(ast.literal_eval(best))


def cmd_package(args) -> None:
    """Build the submission zip exactly as the problem statement lays it out."""
    import shutil
    import zipfile

    root = config.REPO_ROOT
    pkg = root / "business_entity_resolution"
    out_zip = root / f"{args.team}_submission.zip"
    staging = config.WORK_DIR / "package"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "output").mkdir(parents=True)
    (staging / "code").mkdir(parents=True)

    missing = []
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        src = config.OUTPUT_DIR / name
        if not src.is_file():
            missing.append(str(src))
            continue
        shutil.copy2(src, staging / "output" / name)
    if missing:
        raise SystemExit(f"missing output file(s): {', '.join(missing)} — run `ber.cli all` first")

    dest = staging / "code" / "business_entity_resolution"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache",
                                    ".DS_Store", "*.pkl", "*.npy")
    shutil.copytree(pkg / "src", dest / "src", ignore=ignore)
    shutil.copy2(pkg / "README.md", dest / "README.md")
    shutil.copy2(pkg / "requirements.txt", dest / "requirements.txt")
    if (root / "tests").is_dir():
        shutil.copytree(root / "tests", dest / "tests", ignore=ignore)
    doc = root / "Documentation_template.md"
    if doc.is_file():
        shutil.copy2(doc, staging / "Documentation_template.md")
    else:
        raise SystemExit(f"{doc} not found — fill in the methodology template first")
    for extra in ("PROGRESS.md", "RESULTS.md", "run_tests.sh", "finish.sh"):
        if (root / extra).is_file():
            shutil.copy2(root / extra, dest / extra)
    bench = config.REPORT_DIR / "final_report.md"
    if bench.is_file():
        shutil.copy2(bench, dest / "BENCHMARK.md")

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(staging))
    size = out_zip.stat().st_size
    print(f"[package] {out_zip} ({size / 1e6:.1f} MB)")
    with zipfile.ZipFile(out_zip) as z:
        names = z.namelist()
    for need in ("output/matching_results.tsv", "output/candidate_pairs.tsv",
                 "code/business_entity_resolution/README.md",
                 "code/business_entity_resolution/requirements.txt",
                 "Documentation_template.md"):
        assert need in names, f"missing from zip: {need}"
    print(f"[package] {len(names)} files, structure verified")


def cmd_validate_submission(args) -> None:
    import subprocess
    script = config.REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"
    cmd = [sys.executable, str(script),
           "--matching", str(config.OUTPUT_DIR / "matching_results.tsv"),
           "--candidate", str(config.OUTPUT_DIR / "candidate_pairs.tsv"),
           "--test-dir", str(config.TEST_DIR)]
    if args.check_ids:
        cmd.append("--check-ids")
    print(" ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ber", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("prepare", help="parquet caches, corpus statistics, name families")
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_prepare)

    a = sub.add_parser("block", help="candidate generation per country partition")
    a.add_argument("--split", default="train", choices=("train", "test"))
    a.add_argument("--countries", default=None, help="comma separated; default: all in the data")
    a.add_argument("--sample", type=int, default=0, help="queries per country (0 = all)")
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_block)

    a = sub.add_parser("featurize", help="pair features for the candidate table")
    a.add_argument("--split", default="train", choices=("train", "test"))
    a.add_argument("--countries", default=None)
    a.add_argument("--workers", type=int, default=None)
    a.add_argument("--block", type=int, default=4000)
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_featurize)

    a = sub.add_parser("all", help="the whole phase-0 pipeline, end to end")
    a.add_argument("--countries", default=None, help="test countries; default: all")
    a.add_argument("--workers", type=int, default=None)
    a.add_argument("--n-m1", dest="n_m1", type=int, default=None)
    a.add_argument("--n-pre", dest="n_pre", type=int, default=None)
    a.add_argument("--n-m2", dest="n_m2", type=int, default=None)
    a.add_argument("--n-cal", dest="n_cal", type=int, default=None)
    a.add_argument("--n-val", dest="n_val", type=int, default=None)
    a.add_argument("--loco", action="store_true", default=True)
    a.add_argument("--no-loco", dest="loco", action="store_false")
    a.add_argument("--no-test", action="store_true", help="stop after validation")
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_all)

    a = sub.add_parser("package", help="build <team>_submission.zip")
    a.add_argument("--team", default="team", help="team name for the zip filename")
    a.set_defaults(func=cmd_package)

    a = sub.add_parser("validate-submission", help="run the official validator on output/")
    a.add_argument("--check-ids", action="store_true")
    a.set_defaults(func=cmd_validate_submission)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
