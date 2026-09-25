"""Metric tables: blocking quality, matching quality, calibration, errors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from . import config
from .scoring import f05, f05_ceiling, macro_f05_breakdown


def blocking_report(cand: Dict[str, Iterable[str]], gold: Dict[str, Iterable[str]],
                    keys: Sequence[str], pool_size: int) -> Dict[str, float]:
    """Pair recall, full-cluster recall, candidates per S1, reduction ratio, ceiling."""
    r = f05_ceiling(cand, gold, keys)
    total_possible = len(keys) * pool_size
    r["reduction_ratio"] = 1.0 - (r["candidate_pairs"] / total_possible) if total_possible else float("nan")
    r["pool_size"] = pool_size
    return r


def matching_report(pred: Dict[str, Iterable[str]], gold: Dict[str, Iterable[str]],
                    keys: Sequence[str]) -> Dict[str, float]:
    return macro_f05_breakdown(pred, gold, keys)


def as_markdown(rows: List[Dict], columns: Sequence[str] | None = None,
                floatfmt: int = 4) -> str:
    if not rows:
        return "_(no rows)_"
    cols = list(columns) if columns else list(rows[0].keys())
    def fmt(v):
        if isinstance(v, float):
            return "n/a" if not np.isfinite(v) else f"{v:.{floatfmt}f}"
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        return str(v)
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def error_analysis(K: pd.DataFrame, prob: np.ndarray, y: np.ndarray,
                   pred_mask: np.ndarray, q_text: Dict[int, tuple],
                   c_text: Dict[tuple, tuple], n: int = 5) -> Dict[str, List[Dict]]:
    """Top false positives and false negatives with their records attached."""
    fp = np.flatnonzero(pred_mask & (y == 0))
    fn = np.flatnonzero((~pred_mask) & (y == 1))
    fp = fp[np.argsort(-prob[fp])][:n]
    fn = fn[np.argsort(-prob[fn])][:n]

    def rows(idx):
        out = []
        for i in idx:
            s1 = int(K["s1"].iat[i]); src = int(K["src"].iat[i]); cid = int(K["cid"].iat[i])
            qn, qa = q_text.get(s1, ("?", "?"))
            cn, ca = c_text.get((src, cid), ("?", "?"))
            out.append({"s1": f"S1-{s1}", "cand": f"S{src}-{cid}",
                        "p": round(float(prob[i]), 4),
                        "s1_name": qn, "s1_addr": qa, "cand_name": cn, "cand_addr": ca})
        return out
    return {"false_positives": rows(fp), "false_negatives": rows(fn)}


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_default))


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
