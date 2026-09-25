"""Phase 2 — the cross-encoder, and Phase 3 — the LLM judge on the uncertain band.

``bge-reranker-v2-m3`` (568M, Apache-2.0) is the primary matcher: it starts as a
reranker on the same XLM-R backbone as BGE-M3, so it reads Devanagari and
Malayalam.  English-only DeBERTa-v3-large is excluded despite its reputation —
it cannot read a quarter of the Indian S2/S3 names.

Three serialisation choices put cross-attention where the decisions are:

1. A fixed field order with explicit markers and S1 always first, so attention
   aligns name with name and address with address even when the raw address is
   reordered.
2. An extracted ``nums:`` field, so the model sees ``616`` against ``627`` as two
   short adjacent tokens instead of digits buried in a long address.
3. A romanised echo for native scripts.

Training is on the blocker's own output — positives are true pairs inside the
candidate lists, negatives are the other candidates, which are hard by
construction — so the training distribution matches inference.  Two
augmentations matter: shifting a positive's house number by 3–20 and relabelling
it negative (the dominant hard-negative pattern), and the observed
label-preserving noise operators.
"""
from __future__ import annotations

import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from . import config
from .dense import serialise
from .device import check_model, pick_device
from .normalize import house_numbers

JUDGE_INSTRUCTION = (
    "Decide whether both records describe the same business at the same location. "
    "Names may be abbreviated, reordered, transliterated or misspelt, and house "
    "numbers may be zero-padded or truncated. A different house number on the same "
    "street usually means a different business."
)


@dataclass
class CrossEncoderConfig:
    repo: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = 128
    batch_size: int = 64
    lr: float = 2e-5
    epochs: int = 1
    neg_per_pos: int = 3
    warmup_frac: float = 0.05
    seed: int = config.SEED
    device: str | None = None

    @classmethod
    def for_device(cls, device: str | None = None, **kw) -> "CrossEncoderConfig":
        device = device or pick_device()
        if device == "cuda":
            return cls(device=device, **kw)
        small = dict(batch_size=16, max_length=96, device=device)
        small.update(kw)
        return cls(**small)


def pair_text(name: str, addr: str, country: str) -> str:
    return serialise(name, addr, country)


# ------------------------------------------------------------------ augmentation
_NUM = re.compile(r"\d+")


def shift_house_number(addr: str, rng: random.Random,
                       lo: int = 3, hi: int = 20) -> str | None:
    """Move the first house number by 3–20 — the dominant hard-negative pattern.

    29.5% of same-name hard negatives differ from the true pair by 3–20 on the
    same street.  Copying a positive, shifting its number and relabelling it
    negative teaches that boundary directly.
    """
    m = _NUM.search(addr or "")
    if not m:
        return None
    delta = rng.randint(lo, hi) * rng.choice([-1, 1])
    new = max(1, int(m.group()) + delta)
    return addr[:m.start()] + str(new) + addr[m.end():]


def corrupt(text: str, rng: random.Random) -> str:
    """Label-preserving noise: the operators actually observed in the data."""
    r = rng.random()
    if r < 0.2:                                              # zero padding
        m = _NUM.search(text)
        if m:
            return text[:m.start()] + m.group().zfill(6) + text[m.end():]
    if r < 0.4:                                              # token duplication
        parts = text.split()
        if len(parts) > 2:
            i = rng.randrange(len(parts))
            parts.insert(i, parts[i])
            return " ".join(parts)
    if r < 0.6:                                              # character corruption
        if text:
            i = rng.randrange(len(text))
            return text[:i] + rng.choice("aeiou1") + text[i + 1:]
    if r < 0.8:                                              # span deletion (Ditto)
        parts = text.split()
        if len(parts) > 4:
            i = rng.randrange(len(parts) - 2)
            return " ".join(parts[:i] + parts[i + 2:])
    return "-- " + text                                      # placeholder junk


class CrossEncoder:
    """Score (S1, candidate) pairs with a sequence-classification reranker."""

    def __init__(self, cfg: CrossEncoderConfig | None = None):
        self.cfg = cfg or CrossEncoderConfig.for_device()
        check_model(self.cfg.repo)
        self.device = self.cfg.device or pick_device()
        self.model = None
        self.tokenizer = None

    def load(self, weights: Path | None = None) -> "CrossEncoder":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.repo)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.cfg.repo)
        self.model.to(self.device)
        if weights is not None and Path(weights).exists():
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
        return self

    def logits(self, left: Sequence[str], right: Sequence[str],
               batch_size: int | None = None) -> np.ndarray:
        import torch
        bs = batch_size or self.cfg.batch_size
        self.model.eval()
        out: List[np.ndarray] = []
        with torch.no_grad():
            for lo in range(0, len(left), bs):
                enc = self.tokenizer(list(left[lo:lo + bs]), list(right[lo:lo + bs]),
                                     padding=True, truncation=True,
                                     max_length=self.cfg.max_length, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                z = self.model(**enc).logits
                z = z[:, 0] if z.shape[-1] == 1 else z[:, -1]
                out.append(z.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)


def train_cross_encoder(left: Sequence[str], right: Sequence[str], y: np.ndarray,
                        cfg: CrossEncoderConfig | None = None,
                        eval_fn=None, out_path: Path | None = None,
                        verbose: bool = True) -> Tuple[CrossEncoder, Dict]:
    """Binary cross-entropy on the blocker's own candidates.

    The learning rate is deliberately low and training is one to two epochs:
    full fine-tuning trades cross-domain transfer for in-domain gains, and
    cross-domain transfer is exactly what France needs.
    """
    import torch

    cfg = cfg or CrossEncoderConfig.for_device()
    ce = CrossEncoder(cfg).load()
    opt = torch.optim.AdamW(ce.model.parameters(), lr=cfg.lr, weight_decay=0.01)
    n = len(y)
    steps = max(1, math.ceil(n / cfg.batch_size) * cfg.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps,
                                                pct_start=cfg.warmup_frac,
                                                anneal_strategy="linear")
    rng = np.random.default_rng(cfg.seed)
    hist: Dict = {"config": asdict(cfg), "device": ce.device, "n": int(n), "steps": []}
    t0 = time.time()
    ce.model.train()
    step = 0
    best = -1.0
    for epoch in range(cfg.epochs):
        order = rng.permutation(n)
        for lo in range(0, n, cfg.batch_size):
            sel = order[lo:lo + cfg.batch_size]
            enc = ce.tokenizer([left[i] for i in sel], [right[i] for i in sel],
                               padding=True, truncation=True,
                               max_length=cfg.max_length, return_tensors="pt")
            enc = {k: v.to(ce.device) for k, v in enc.items()}
            z = ce.model(**enc).logits
            z = z[:, 0] if z.shape[-1] == 1 else z[:, -1]
            target = torch.tensor(y[sel], dtype=z.dtype, device=ce.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ce.model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if verbose and step % 20 == 0:
                print(f"  [ce] step {step}/{steps} loss={float(loss):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        if eval_fn is not None:
            ce.model.eval()
            score = float(eval_fn(ce))
            ce.model.train()
            hist["steps"].append({"epoch": epoch, "eval": score})
            if verbose:
                print(f"  [ce] epoch {epoch}: eval={score:.4f}", flush=True)
            if score > best:
                best = score
                if out_path is not None:
                    ce.save(out_path)
    if out_path is not None and best < 0:
        ce.save(out_path)
    hist["best_eval"] = best
    hist["seconds"] = round(time.time() - t0, 1)
    return ce, hist


def uncertain_band(prob: np.ndarray, lo: float = 0.1, hi: float = 0.9) -> np.ndarray:
    """Rows the LLM judge should see — typically 5–10% of pairs."""
    p = np.asarray(prob)
    return (p >= lo) & (p <= hi)
