"""End-to-end drivers: one function per phase-0 stage, all resumable.

Order::

    prepare -> block(train) -> fit_preranker -> prune(train) -> featurize(train)
            -> fit_stack -> validate -> block(test) -> prune(test)
            -> featurize(test) -> infer -> write TSVs

Every stage writes its output under ``work/`` and skips itself when that output
already exists, so a long run can be interrupted and resumed.
"""
from __future__ import annotations

import gc
import json
import time
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config
from .corpus import build_roles, roles_path
from .data import countries as data_countries, load_partition
from .featurize import DEFAULT_BLOCK_S1, featurize_partition
from .io_utils import read_s1_ids, write_candidates, write_matching
from .labels import GroundTruth
from .preranker import model_path as preranker_path
from .pipeline import (Selection, apply_decision, plan_selection, search_decision)
from .report import as_markdown, blocking_report, matching_report, save_json
from .retrieval import (BlockingConfig, cand_dir, candidates_as_dict,
                        load_candidates, retrieve_partition)
from .scoring import f05_ceiling, macro_f05
from .splits import build_name_families, fold_ids, reweighted_score, sample_by_family
from .train import Stack, fit_stack

#: How many S1 entities play each role, per country.  Sized so the whole training
#: blocking pass (640k queries over both countries) fits in about 45 minutes on this
#: machine while still giving M1 ~3M labelled pairs and a validation fold whose
#: macro-F_0.5 standard error is around 0.001.
DEFAULT_SIZES = dict(n_m1=120_000, n_pre=50_000, n_m2=50_000, n_cal=30_000,
                     n_val=120_000)


# ------------------------------------------------------------------ planning
def make_selection(fam: pd.DataFrame, sizes: Dict[str, int] | None = None) -> Selection:
    """Who trains what.  ``n_val=0`` means the whole validation fold.

    Sampling is by **name family**, never by individual entity: fold assignment
    is already a function of the family, so a whole family lives in one fold and
    the exclusivity features see the same contest they will see at test time.
    """
    sizes = {**DEFAULT_SIZES, **(sizes or {})}
    sel = Selection()
    for c in sorted(fam["country"].astype(str).unique()):
        ctry = fam["country"].to_numpy() == c
        fold = fam["fold"].to_numpy()
        m1_mask = ctry & np.isin(fold, (1, 2))
        pre_mask = ctry & (fold == 3)
        m2_mask = ctry & (fold == 4)
        val_mask = ctry & (fold == 0)
        sel.m1[c] = sample_by_family(fam, m1_mask, sizes["n_m1"], config.SEED)
        sel.pre[c] = sample_by_family(fam, pre_mask, sizes["n_pre"], config.SEED + 4)
        meta = sample_by_family(fam, m2_mask, sizes["n_m2"] + sizes["n_cal"],
                                config.SEED + 1)
        # Split fold 4 between M2 and calibration **by family**, not by id: two
        # S1 entities sharing a name must not straddle the two, or the isotonic
        # fit is calibrating a model that has already seen their look-alike.
        # The union is unchanged, so this does not disturb any cached blocking.
        sel.m2[c], sel.cal[c] = _split_by_family(
            fam, meta, sizes["n_m2"] / max(sizes["n_m2"] + sizes["n_cal"], 1))
        sel.val[c] = (np.sort(fam["id"].to_numpy(dtype=np.int32)[val_mask])
                      if not sizes["n_val"]
                      else sample_by_family(fam, val_mask, sizes["n_val"], config.SEED + 2))
    return sel


def _split_by_family(fam: pd.DataFrame, ids: np.ndarray,
                     first_share: float) -> Tuple[np.ndarray, np.ndarray]:
    """Split ``ids`` in two, keeping every name family wholly on one side."""
    if not len(ids):
        return ids, ids
    lut = dict(zip(fam["id"].to_numpy(dtype=np.int32).tolist(),
                   fam["family"].to_numpy().tolist()))
    cut = int(first_share * 1000)
    first = np.fromiter(
        (zlib.crc32(str(lut.get(int(i), i)).encode("utf-8")) % 1000 < cut for i in ids),
        dtype=bool, count=len(ids))
    return np.sort(ids[first]), np.sort(ids[~first])


def selection_queries(sel: Selection, countries: Sequence[str]) -> Dict[str, np.ndarray]:
    return {c: sel.all_ids(c) for c in countries}


# ------------------------------------------------------------------ stages
def stage_block(split: str, bcfg: BlockingConfig, countries: Sequence[str],
                q_by_country: Dict[str, np.ndarray] | None,
                force: bool = False) -> None:
    roles = build_roles(split)
    for c in countries:
        retrieve_partition(split, c, bcfg, roles,
                           q_ids=None if q_by_country is None else q_by_country.get(c),
                           force=force)
        gc.collect()


def stage_fit_preranker(bcfg: BlockingConfig, countries: Sequence[str],
                        sel: Selection, gt: GroundTruth,
                        max_rows_per_country: int = 12_000_000,
                        force: bool = False):
    """Train the last blocking filter on the raw union of the training folds."""
    from .model import TrainedModel
    from .preranker import train_preranker
    path = preranker_path(bcfg)
    if path.exists() and not force:
        return TrainedModel.load(path)
    frames = []
    for c in countries:
        # fold 3 only: the pre-ranker's score is an M1 feature, so it must be
        # out-of-sample everywhere M1, M2, calibration and validation look
        df = load_candidates("train", c, bcfg, pruned=False, s1_ids=sel.pre[c])
        if len(df) > max_rows_per_country:
            uniq = np.unique(df["s1"].to_numpy())
            frac = max_rows_per_country / len(df)
            rng = np.random.default_rng(config.SEED)
            keep = uniq[rng.random(len(uniq)) < frac]
            df = df[np.isin(df["s1"].to_numpy(), keep)]
        frames.append(df)
        gc.collect()
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    y = gt.label(df["s1"].to_numpy(), df["src"].to_numpy(), df["cid"].to_numpy())
    uniq = np.unique(df["s1"].to_numpy())
    rng = np.random.default_rng(config.SEED + 5)
    hold_ids = uniq[rng.random(len(uniq)) < 0.15]
    hold = np.isin(df["s1"].to_numpy(), hold_ids)
    model = train_preranker(df, y, hold)
    model.save(path)
    print(f"[preranker] trained on {len(df):,} rows "
          f"(pos {y.mean():.3f}), iters={model.booster.best_iteration}", flush=True)
    del df, y
    gc.collect()
    return model


def stage_prune(split: str, bcfg: BlockingConfig, countries: Sequence[str],
                model, force: bool = False) -> None:
    from .preranker import prerank_partition
    for c in countries:
        prerank_partition(split, c, bcfg, model, force=force)
        gc.collect()


def stage_featurize(split: str, bcfg: BlockingConfig, countries: Sequence[str],
                    n_workers: int | None = None,
                    block_s1: int = DEFAULT_BLOCK_S1,
                    force: bool = False) -> None:
    rp = roles_path(split)
    for c in countries:
        featurize_partition(split, c, bcfg, rp, block_s1=block_s1,
                            n_workers=n_workers, force=force)
        gc.collect()


# ------------------------------------------------------------------ reporting
def blocking_metrics(split: str, bcfg: BlockingConfig, countries: Sequence[str],
                     gt_dict: Dict[str, List[str]], sel: Selection,
                     pruned: bool) -> List[Dict]:
    rows = []
    for c in countries:
        ids = sel.val[c]
        df = load_candidates(split, c, bcfg, columns=["s1", "src", "cid"],
                             pruned=pruned, s1_ids=ids)
        cand = candidates_as_dict(df)
        keys = [f"S1-{i}" for i in ids]
        pool = len(load_partition(split, "S2", c)) + len(load_partition(split, "S3", c))
        r = blocking_report(cand, gt_dict, keys, pool)
        r["country"] = c
        r["stage"] = "pruned" if pruned else "raw union"
        rows.append(r)
        del df, cand
        gc.collect()
    return rows


# ------------------------------------------------------------------ validation
def validate(bcfg: BlockingConfig, stack: Stack, countries: Sequence[str],
             sel: Selection, gt_dict: Dict[str, List[str]],
             split: str = "train", verbose: bool = True) -> Dict:
    """Score the held-out fold per country, search the decision rule, report."""
    from .infer import score_country

    per_country: Dict[str, Dict] = {}
    scored: Dict[str, Tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for c in countries:
        K, p, pe = score_country(split, c, bcfg, stack, verbose=verbose)
        m = np.isin(K["s1"].to_numpy(), sel.val[c])
        scored[c] = (K.loc[m].reset_index(drop=True), p[m], pe[m])
        del K, p, pe
        gc.collect()

    out: Dict = {"per_country": {}, "decision": {}}
    for c in countries:
        K, p, pe = scored[c]
        keys = [f"S1-{i}" for i in sel.val[c]]
        rule, score, table = search_decision(K, pe, gt_dict, keys)
        pred = apply_decision(K, pe, rule)
        rep = matching_report(pred, gt_dict, keys)
        rep["rule"] = str(rule)
        rep["country"] = c
        per_country[c] = rep
        out["decision"][c] = {str(k): round(v, 5) for k, v in table.items()}
        out["per_country"][c] = rep
        if verbose:
            print(f"[validate] {c}: macro F0.5 = {rep['macro_f05']:.4f} "
                  f"(rule {rule}, P={rep['micro_precision']:.3f} "
                  f"R={rep['micro_recall']:.3f})", flush=True)
    out["scored"] = scored
    return out


def run_loco(bcfg: BlockingConfig, countries: Sequence[str], sel: Selection,
             gt: GroundTruth, gt_dict: Dict[str, List[str]],
             verbose: bool = True) -> Dict:
    """Leave-one-country-out: the only honest proxy for an unseen country.

    Train the whole stack on one country and score another.  The gap to the
    in-country score is the estimate of what France will cost.
    """
    from .infer import score_country
    out: Dict[str, Dict] = {}
    for train_c in countries:
        for eval_c in countries:
            if train_c == eval_c:
                continue
            stack = fit_stack("train", [train_c], bcfg, sel, gt, verbose=False)
            K, p, pe = score_country("train", eval_c, bcfg, stack, verbose=False)
            m = np.isin(K["s1"].to_numpy(), sel.val[eval_c])
            Kv, pev = K.loc[m].reset_index(drop=True), pe[m]
            keys = [f"S1-{i}" for i in sel.val[eval_c]]
            rule, score, _ = search_decision(Kv, pev, gt_dict, keys)
            out[f"{train_c}->{eval_c}"] = {"macro_f05": round(float(score), 5),
                                           "rule": str(rule),
                                           "n": len(keys)}
            if verbose:
                print(f"[loco] train {train_c} -> score {eval_c}: "
                      f"{score:.4f} ({rule})", flush=True)
            del stack, K, p, pe, Kv, pev
            gc.collect()
    vals = [v["macro_f05"] for v in out.values()]
    out["mean"] = {"macro_f05": round(float(np.mean(vals)), 5)} if vals else {}
    return out


# ------------------------------------------------------------------ outputs
def write_outputs(bcfg: BlockingConfig, stack: "Stack", rule: Tuple,
                  countries: Sequence[str], out_dir: Path | None = None,
                  verbose: bool = True) -> Dict:
    """Score the test split and write both TSVs in ``test_source1.tsv`` row order.

    Candidates are carried as numpy arrays, never as a dict of string lists: the
    test candidate set is ~47M ids, which costs a few hundred megabytes as
    ``(int32, int8, int32, bool)`` and several gigabytes as Python strings.
    """
    from .infer import score_country
    from .pipeline import decision_mask

    out_dir = out_dir or config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: Dict = {"countries": {}, "rule": str(rule)}

    s1_parts, src_parts, cid_parts, keep_parts = [], [], [], []
    for c in countries:
        K, p, pe = score_country("test", c, bcfg, stack, verbose=verbose)
        keep = decision_mask(K, pe, rule)
        s1_parts.append(K["s1"].to_numpy().astype(np.int32))
        src_parts.append(K["src"].to_numpy().astype(np.int8))
        cid_parts.append(K["cid"].to_numpy().astype(np.int32))
        keep_parts.append(keep)
        stats["countries"][c] = {"pairs_scored": int(len(K)),
                                 "ids_predicted": int(keep.sum()),
                                 "s1": int(K["s1"].nunique())}
        del K, p, pe, keep
        gc.collect()

    s1 = np.concatenate(s1_parts); src = np.concatenate(src_parts)
    cid = np.concatenate(cid_parts); keep = np.concatenate(keep_parts)
    del s1_parts, src_parts, cid_parts, keep_parts
    gc.collect()
    order = np.argsort(s1, kind="stable")
    s1, src, cid, keep = s1[order], src[order], cid[order], keep[order]
    del order
    gc.collect()

    cr = _write_rows(out_dir / "candidate_pairs.tsv",
                     ["source1_entity_id", "candidate_entity_ids"],
                     s1, src, cid, None)
    mr = _write_rows(out_dir / "matching_results.tsv",
                     ["source1_entity_id", "matched_entity_ids"],
                     s1, src, cid, keep)
    stats["candidate_pairs.tsv"] = dict(zip(("rows", "non_empty", "ids"), cr))
    stats["matching_results.tsv"] = dict(zip(("rows", "non_empty", "ids"), mr))
    if verbose:
        print(f"[output] matching_results.tsv: {mr[0]:,} rows, {mr[1]:,} non-empty, "
              f"{mr[2]:,} ids", flush=True)
        print(f"[output] candidate_pairs.tsv:  {cr[0]:,} rows, {cr[1]:,} non-empty, "
              f"{cr[2]:,} ids", flush=True)
    return stats


def _write_rows(path: Path, header: Sequence[str], s1: np.ndarray, src: np.ndarray,
                cid: np.ndarray, keep: np.ndarray | None) -> Tuple[int, int, int]:
    """One row per test S1 in file order, LF endings, no quoting, no spaces."""
    starts = np.searchsorted(s1, np.unique(s1))
    uniq = s1[starts]
    ends = np.r_[starts[1:], len(s1)]
    rows = non_empty = total = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(header) + "\n")
        with open(config.FILES[("test", "S1")], encoding="utf-8") as src_fh:
            src_fh.readline()
            for line in src_fh:
                if not line.strip():
                    continue
                eid = line.split("\t", 1)[0].strip()
                rows += 1
                num = int(eid[3:])
                pos = int(np.searchsorted(uniq, num))
                ids: List[str] = []
                if pos < len(uniq) and uniq[pos] == num:
                    a, b = int(starts[pos]), int(ends[pos])
                    seen = set()
                    for j in range(a, b):
                        if keep is not None and not keep[j]:
                            continue
                        key = (int(src[j]), int(cid[j]))
                        if key in seen:
                            continue
                        seen.add(key)
                        ids.append(f"S{src[j]}-{cid[j]}")
                if ids:
                    non_empty += 1
                    total += len(ids)
                fh.write(f"{eid}\t{','.join(ids)}\n")
    return rows, non_empty, total
