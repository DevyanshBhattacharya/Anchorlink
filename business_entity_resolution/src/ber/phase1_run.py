"""``python -m ber.phase1_run`` — the one command that runs Phase 1 end to end.

Defaults are sized for this Mac (a slice, a small multilingual encoder, a short
fine-tune on MPS).  ``--scale aws`` switches to the single-24 GB-GPU config from
the research doc: BGE-M3, 96 clusters per batch, the full training pool.

**The run is split into two processes on purpose.**  The sparse baseline uses
``sparse_dot_topn``'s OpenMP runtime and the dense stage uses PyTorch's.  Holding
both in one process on macOS either serialises the sparse product (a measured
25x slowdown) or deadlocks it, so ``--stage sparse`` writes its result to disk
and ``--stage dense`` picks it up in a fresh interpreter.  ``--stage all``
(the default) simply runs the two as subprocesses.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np

from . import config
from .dense import Encoder, EncoderConfig
from .dense_train import (ClusterSet, TrainConfig, build_cluster_set,
                          collisions_from_candidates, train_bi_encoder)
from .device import device_info, pick_device
from .io_utils import read_ground_truth
from .phase1 import (build_slice, dense_slice_candidates, recall_at_k,
                     sparse_slice_candidates, union)
from .report import save_json
from .retrieval import DEFAULT_BLOCKING

SCALES = {
    # what fits here: a slice, a 118M MIT encoder, a short fine-tune on MPS
    "mac": dict(repo="intfloat/multilingual-e5-small", n_clusters=6000,
                pool_size=80_000, train_clusters=6000, max_steps=150,
                clusters_per_batch=24, micro_batch=24, max_length=64,
                batch_size=128, k=30, pooling="mean"),
    # the doc's single-24 GB-GPU plan
    "aws": dict(repo="BAAI/bge-m3", n_clusters=200_000, pool_size=4_000_000,
                train_clusters=2_000_000, max_steps=0,
                clusters_per_batch=96, micro_batch=64, max_length=96,
                batch_size=256, k=30, pooling="cls"),
}


def _cache(scale: str, country: str, fold: int, n_clusters: int, pool: int) -> Path:
    return (config.CACHE_DIR /
            f"phase1_slice_{scale}_{country}_f{fold}_q{n_clusters}_p{pool}.pkl")


def run_sparse(args, S) -> None:
    """Stage 1: build the slice and the sparse baseline.  No torch in this process."""
    import pickle

    report: Dict = {"scale": args.scale, "country": args.country, "settings": S}
    t0 = time.time()
    sl = build_slice(args.country, S["n_clusters"], S["pool_size"], fold=args.fold)
    report["slice"] = sl.stats()
    print(f"[phase1] slice: {report['slice']} in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    sparse = sparse_slice_candidates(sl, DEFAULT_BLOCKING, S["k"])
    report["sparse"] = recall_at_k(sparse, sl.truth)
    report["sparse"]["seconds"] = round(time.time() - t0, 1)
    print(f"[phase1] sparse union: {report['sparse']}", flush=True)

    path = _cache(args.scale, args.country, args.fold, S["n_clusters"], S["pool_size"])
    with open(path, "wb") as fh:
        pickle.dump({"slice": sl, "sparse": sparse, "report": report}, fh, protocol=4)
    print(f"[phase1] stage sparse done -> {path}", flush=True)


def run_dense(args, S) -> None:
    """Stage 2: the encoder.  Torch owns the OpenMP runtime in this process."""
    import pickle

    path = _cache(args.scale, args.country, args.fold, S["n_clusters"], S["pool_size"])
    if not path.exists():
        raise SystemExit(f"{path} missing — run `--stage sparse` first")
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    sl, sparse, report = blob["slice"], blob["sparse"], blob["report"]
    report["device"] = device_info()
    print(f"[phase1] device={report['device']}", flush=True)

    enc = Encoder(EncoderConfig(repo=S["repo"], max_length=S["max_length"],
                                batch_size=S["batch_size"], pooling=S["pooling"])).load()
    dense, timing = dense_slice_candidates(enc, sl, S["k"], S["batch_size"])
    report["dense_zero_shot"] = {**recall_at_k(dense, sl.truth), **timing}
    report["union_zero_shot"] = recall_at_k(union(sparse, dense), sl.truth)
    print(f"[phase1] dense zero-shot: {report['dense_zero_shot']}", flush=True)
    print(f"[phase1] union zero-shot: {report['union_zero_shot']}", flush=True)

    if args.train:
        from .retrieval import load_candidates
        gt = read_ground_truth()
        q_num = np.array([int(q[3:]) for q in sl.q_ids], dtype=np.int32)
        data = build_cluster_set("train", [args.country], gt, {args.country: q_num},
                                 max_distractors_per_country=S["pool_size"] // 4)
        cluster_of_s1, cluster_of_record = {}, {}
        for cid, rows in data.clusters.items():
            for r in rows:
                src, num = data.record_key[r]
                if src == 1:
                    cluster_of_s1[num] = cid
                else:
                    cluster_of_record[(src, num)] = cid
        try:
            cand = load_candidates("train", args.country, DEFAULT_BLOCKING,
                                   columns=["s1", "src", "cid"], pruned=False)
            cand = cand[np.isin(cand["s1"].to_numpy(), q_num)]
            collisions = collisions_from_candidates(cand, cluster_of_s1, cluster_of_record)
        except FileNotFoundError:
            print("[phase1] no candidate table yet — batching without collisions",
                  flush=True)
            collisions = {}
        del enc
        gc.collect()
        cfg = TrainConfig(repo=S["repo"], max_length=S["max_length"],
                          clusters_per_batch=S["clusters_per_batch"],
                          micro_batch=S["micro_batch"], max_steps=S["max_steps"],
                          eval_every=0)
        out_w = config.MODEL_DIR / f"biencoder_{args.scale}_{args.country}.pt"
        enc, hist = train_bi_encoder(data, collisions, cfg, out_path=out_w)
        report["training"] = hist
        dense2, timing2 = dense_slice_candidates(enc, sl, S["k"], S["batch_size"])
        report["dense_tuned"] = {**recall_at_k(dense2, sl.truth), **timing2}
        report["union_tuned"] = recall_at_k(union(sparse, dense2), sl.truth)
        print(f"[phase1] dense tuned: {report['dense_tuned']}", flush=True)
        print(f"[phase1] union tuned: {report['union_tuned']}", flush=True)

    out = Path(args.out) if args.out else config.REPORT_DIR / f"phase1_{args.scale}.json"
    save_json(out, report)
    print(f"[phase1] -> {out}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=sorted(SCALES), default="mac")
    ap.add_argument("--country", default="India")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--train", action="store_true", help="fine-tune, not just zero-shot")
    ap.add_argument("--stage", choices=("all", "sparse", "dense"), default="all")
    ap.add_argument("--n-clusters", dest="n_clusters", type=int, default=None,
                    help="override the preset's query-cluster count")
    ap.add_argument("--pool-size", dest="pool_size", type=int, default=None,
                    help="override the preset's pool size")
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    S = dict(SCALES[args.scale])
    for k in ("n_clusters", "pool_size", "max_steps"):
        v = getattr(args, k)
        if v is not None:
            S[k] = v
    print(f"[phase1] scale={args.scale} country={args.country} stage={args.stage}",
          flush=True)

    if args.stage == "sparse":
        return run_sparse(args, S)
    if args.stage == "dense":
        return run_dense(args, S)

    import subprocess
    import sys as _sys
    base = [_sys.executable, "-u", "-m", "ber.phase1_run",
            "--scale", args.scale, "--country", args.country,
            "--fold", str(args.fold)]
    for flag, key in (("--n-clusters", "n_clusters"), ("--pool-size", "pool_size"),
                      ("--max-steps", "max_steps")):
        v = getattr(args, key)
        if v is not None:
            base += [flag, str(v)]
    if args.train:
        base.append("--train")
    if args.out:
        base += ["--out", args.out]
    for stage in ("sparse", "dense"):
        rc = subprocess.call(base + ["--stage", stage])
        if rc != 0:
            raise SystemExit(f"phase1 stage {stage} failed with code {rc}")


if __name__ == "__main__":
    main()
