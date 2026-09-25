"""``python -m ber.phase2_run`` — the cross-encoder phase, end to end.

The cross-encoder is trained on the blocker's **own** candidates, so the training
distribution matches inference: positives are the true pairs inside the candidate
lists, negatives are the other candidates, which are hard by construction.  Two
augmentations are added on top, both aimed at the boundary the data says decides
the hard cases: a copy of a positive with its house number shifted 3–20 and the
label flipped to negative, and label-preserving noise (zero padding, duplication,
span deletion, placeholder junk).

The gain is measured the way it will count: the cross-encoder's logit becomes one
more column of the meta model, M2 is retrained with it, and the validation macro
F_0.5 is compared with and without.  Anything that does not beat the previous best
on the cross-country fold is dropped.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config
from .cross_encoder import (CrossEncoder, CrossEncoderConfig, corrupt, pair_text,
                            shift_house_number, train_cross_encoder)
from .data import load_partition
from .device import device_info
from .featurize import load_features
from .io_utils import read_ground_truth
from .labels import GroundTruth
from .model import fit_calibrator, train_lgb
from .pipeline import STAGE2_FEATURES, add_meta, search_decision, sort_by_s1
from .report import save_json
from .retrieval import DEFAULT_BLOCKING, load_candidates
from .run import make_selection
from .scoring import macro_f05
from .splits import build_name_families
from .train import Stack

#: Sized from a measured 31.6 pairs/s for inference on Apple MPS (568M
#: cross-encoder, 96 tokens) and roughly a third of that for a training step.
#: Scoring the full test candidate set at that rate would take over two weeks,
#: which is why Phase 2 at full scale is AWS work and this preset measures the
#: *gain* on a subsample instead.
SCALES = {
    "mac": dict(ce_train_pairs=6_000, meta_pairs=8_000, eval_pairs=12_000,
                batch_size=16, max_length=96, epochs=1),
    "aws": dict(ce_train_pairs=4_000_000, meta_pairs=2_000_000, eval_pairs=0,
                batch_size=64, max_length=128, epochs=1),
}


def _record_text(split: str, country: str) -> Tuple[Dict, Dict]:
    """``(s1 lookup, candidate lookup)`` from entity number to (name, addr)."""
    s1 = load_partition(split, "S1", country)
    q = {int(i): (n, a) for i, n, a in zip(s1.ids, s1.names, s1.addrs)}
    c: Dict[Tuple[int, int], Tuple[str, str]] = {}
    for src in (2, 3):
        p = load_partition(split, f"S{src}", country)
        for i, n, a in zip(p.ids, p.names, p.addrs):
            c[(src, int(i))] = (n, a)
        del p
        gc.collect()
    del s1
    return q, c


def build_pairs(K: pd.DataFrame, y: np.ndarray, country: str, split: str,
                limit: int, augment: bool, seed: int = config.SEED
                ) -> Tuple[List[str], List[str], np.ndarray]:
    """Serialise candidate pairs, optionally with the number-shift augmentation."""
    rng = random.Random(seed)
    q_txt, c_txt = _record_text(split, country)
    idx = np.arange(len(K))
    if limit and len(idx) > limit:
        rs = np.random.default_rng(seed)
        pos = idx[y == 1]
        neg = idx[y == 0]
        n_pos = min(len(pos), max(1, limit // 4))
        n_neg = min(len(neg), limit - n_pos)
        idx = np.sort(np.concatenate([rs.choice(pos, n_pos, replace=False),
                                      rs.choice(neg, n_neg, replace=False)]))
    left, right, label = [], [], []
    s1a = K["s1"].to_numpy(); srca = K["src"].to_numpy(); cida = K["cid"].to_numpy()
    for i in idx:
        qn, qa = q_txt.get(int(s1a[i]), ("", ""))
        cn, ca = c_txt.get((int(srca[i]), int(cida[i])), ("", ""))
        left.append(pair_text(qn, qa, country))
        right.append(pair_text(cn, ca, country))
        label.append(float(y[i]))
        if augment and y[i] == 1:
            shifted = shift_house_number(ca, rng)
            if shifted is not None:                      # label-changing
                left.append(pair_text(qn, qa, country))
                right.append(pair_text(cn, shifted, country))
                label.append(0.0)
            if rng.random() < 0.3:                       # label-preserving
                left.append(pair_text(qn, qa, country))
                right.append(pair_text(corrupt(cn, rng), ca, country))
                label.append(1.0)
    del q_txt, c_txt
    gc.collect()
    return left, right, np.asarray(label, dtype=np.float32)


def _subsample_groups(K: pd.DataFrame, mask: np.ndarray, n_pairs: int,
                      seed: int = config.SEED) -> np.ndarray:
    """Row indices for whole S1 groups, up to roughly ``n_pairs`` rows.

    Whole groups, because macro F_0.5 is only defined per entity — half an
    entity's candidates would make the score meaningless.
    """
    idx = np.flatnonzero(mask)
    if not n_pairs or len(idx) <= n_pairs:
        return idx
    s1 = K["s1"].to_numpy()[idx]
    uniq = np.unique(s1)
    rng = np.random.default_rng(seed)
    keep = uniq[rng.permutation(len(uniq))]
    per = len(idx) / max(len(uniq), 1)
    keep = keep[:max(1, int(n_pairs / max(per, 1e-9)))]
    return idx[np.isin(s1, keep)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=sorted(SCALES), default="mac")
    ap.add_argument("--country", default="India")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-train", action="store_true",
                    help="score with the released weights instead of fine-tuning")
    args = ap.parse_args(argv)
    S = SCALES[args.scale]
    bcfg = DEFAULT_BLOCKING
    report: Dict = {"scale": args.scale, "country": args.country,
                    "settings": S, "device": device_info()}

    fam = build_name_families("train")
    sel = make_selection(fam)
    gt = GroundTruth()
    gt_dict = read_ground_truth()

    X, K = load_features("train", args.country, bcfg)
    X, K = sort_by_s1(X, K)
    y = gt.label(K["s1"].to_numpy(), K["src"].to_numpy(), K["cid"].to_numpy())
    s1 = K["s1"].to_numpy()
    in_m2 = np.isin(s1, sel.m2[args.country])
    in_val = np.isin(s1, sel.val[args.country])
    report["rows"] = {"all": int(len(K)), "m2": int(in_m2.sum()), "val": int(in_val.sum())}
    print(f"[phase2] rows {report['rows']}", flush=True)

    meta_idx = _subsample_groups(K, in_m2, S["meta_pairs"])
    val_idx = _subsample_groups(K, in_val, S["eval_pairs"], seed=config.SEED + 1)
    print(f"[phase2] meta rows {len(meta_idx):,}  val rows {len(val_idx):,}", flush=True)

    # ---- the cross-encoder -------------------------------------------
    t0 = time.time()
    if args.skip_train:
        ce = CrossEncoder(CrossEncoderConfig.for_device(
            batch_size=S["batch_size"], max_length=S["max_length"])).load()
        hist = {"note": "released weights, no fine-tuning"}
    else:
        ce_idx = _subsample_groups(K, in_m2, S["ce_train_pairs"], seed=config.SEED + 2)
        left, right, lab = build_pairs(K.iloc[ce_idx].reset_index(drop=True), y[ce_idx],
                                       args.country, "train", 0, augment=True)
        print(f"[phase2] fine-tuning on {len(lab):,} pairs (pos {lab.mean():.3f}) "
              f"built in {time.time()-t0:.0f}s", flush=True)
        cfg = CrossEncoderConfig.for_device(batch_size=S["batch_size"],
                                            max_length=S["max_length"],
                                            epochs=S["epochs"])
        out_w = config.MODEL_DIR / f"crossencoder_{args.scale}_{args.country}.pt"
        ce, hist = train_cross_encoder(left, right, lab, cfg, out_path=out_w)
        del left, right, lab
        gc.collect()
    report["training"] = hist

    def score(idx: np.ndarray, tag: str) -> np.ndarray:
        L, R, _ = build_pairs(K.iloc[idx].reset_index(drop=True), y[idx],
                              args.country, "train", 0, augment=False)
        t = time.time()
        z = ce.logits(L, R)
        el = time.time() - t
        print(f"[phase2] scored {tag}: {len(z):,} pairs in {el:.0f}s "
              f"({len(z)/max(el,1e-9):.1f}/s)", flush=True)
        report.setdefault("scoring", {})[tag] = {
            "pairs": int(len(z)), "seconds": round(el, 1),
            "pairs_per_s": round(len(z) / max(el, 1e-9), 1)}
        return z

    z_meta = score(meta_idx, "meta")
    z_val = score(val_idx, "val")
    np.save(config.MODEL_DIR / f"ce_logits_val_{args.scale}_{args.country}.npy", z_val)

    # ---- does the logit earn its place in the meta model? -------------
    stack_dir = None
    from .features import FEATURE_VERSION
    for d in sorted(config.MODEL_DIR.glob(
            f"{bcfg.fingerprint()}_f{FEATURE_VERSION}__*")):
        if (d / "m1.pkl").exists():
            stack_dir = d
    if stack_dir is None:
        print("[phase2] no trained stack yet; skipping the meta ablation", flush=True)
    else:
        stack = Stack.load(stack_dir)
        keys = [f"S1-{i}" for i in np.unique(K["s1"].to_numpy()[val_idx])]

        def meta_matrix(idx, z):
            Xi, Ki = X[idx], K.iloc[idx].reset_index(drop=True)
            p1 = stack.m1.raw_predict(Xi)
            return add_meta(Xi, Ki, p1), Ki, np.asarray(z, dtype=np.float32)

        Xm, Km, zm = meta_matrix(meta_idx, z_meta)
        Xv, Kv, zv = meta_matrix(val_idx, z_val)
        names = list(STAGE2_FEATURES)

        rows = []
        for label, tr, va, cols in (
                ("without cross-encoder", Xm, Xv, names),
                ("with cross-encoder", np.hstack([Xm, zm[:, None]]),
                 np.hstack([Xv, zv[:, None]]), names + ["ce_logit"])):
            m2 = train_lgb(tr, y[meta_idx], cols, num_boost_round=400,
                           early_stopping=0)
            p = np.clip(m2.raw_predict(va), 1e-6, 1 - 1e-6)
            rule, best, _ = search_decision(Kv, p, gt_dict, keys)
            rows.append({"variant": label, "features": len(cols),
                         "macro_f05": round(float(best), 5), "rule": str(rule)})
            print(f"[phase2] {label}: macro F0.5 = {best:.5f} ({rule})", flush=True)
        report["meta_ablation"] = rows
        delta = rows[1]["macro_f05"] - rows[0]["macro_f05"]
        report["gain"] = round(delta, 5)
        report["kept"] = bool(delta > 0)
        print(f"[phase2] cross-encoder gain on this subsample: {delta:+.5f} "
              f"-> {'keep' if delta > 0 else 'drop'}", flush=True)

    save_json(Path(args.out) if args.out else
              config.REPORT_DIR / f"phase2_{args.scale}_{args.country}.json", report)
    print("[phase2] done", flush=True)


if __name__ == "__main__":
    main()
