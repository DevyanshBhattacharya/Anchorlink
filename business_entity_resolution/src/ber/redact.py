"""Mask challenge records before they are written anywhere that gets published.

The error analysis is only useful if you can see *why* a pair is hard, but the
records themselves are the organisers' data and do not belong in a public repo.
The masking keeps every structural property the analysis depends on and throws
away the identity:

* each distinct content word becomes a stable placeholder (``W1``, ``W2``, ...)
  within one example, so "dropped a word" is still visible across the four
  strings of that example;
* generic vocabulary — legal forms, street types, directional words — is kept,
  because it is English, not data, and it is what the pattern is *about*;
* digits are kept, because the house-number relation is half the analysis.

So the worked example still reads exactly as it should::

    S1:   W1 W2 Private Limited   ·  36/2342 W5 W6 W7
    cand: W1 W2 Limited           ·  36/2342 W5 W6 W7

— the reader sees the dropped ``Private`` without ever seeing the business.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

#: Generic vocabulary, kept verbatim: it carries the pattern, not the identity.
KEEP = {
    # legal forms, the thing most patterns are about
    "private", "limited", "ltd", "llp", "llc", "inc", "incorporated", "corp",
    "corporation", "co", "company", "pvt", "plc", "lp", "pllc",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei", "ets", "associes",
    # street and address vocabulary
    "road", "rd", "street", "st", "drive", "dr", "avenue", "ave", "lane", "ln",
    "boulevard", "blvd", "highway", "hwy", "marg", "nagar", "colony", "sector",
    "phase", "block", "plot", "floor", "flat", "door", "house", "building",
    "tower", "complex", "market", "rue", "bis", "allee", "place", "square",
    "no", "num", "number", "opp", "opposite", "near", "behind", "next", "above",
    "north", "south", "east", "west", "new", "old", "main", "cross",
    "and", "the", "of", "de", "du", "la", "le", "des",
}

#: Alphanumeric runs, so "2Nd" stays one token instead of "2" + "Nd".
_TOKEN = re.compile("[0-9A-Za-z\u00c0-\u024f\u0900-\u0dff]+")


class Masker:
    """Stable placeholders within one example, fresh for the next."""

    def __init__(self) -> None:
        self._map: Dict[str, str] = {}

    def reset(self) -> None:
        self._map.clear()

    def word(self, w: str) -> str:
        if w.isdigit():
            return w                      # house numbers carry half the analysis
        low = w.lower()
        if low in KEEP:
            return w
        if low not in self._map:
            self._map[low] = f"W{len(self._map) + 1}"
        return self._map[low]

    def text(self, s: str) -> str:
        if not s:
            return s
        return _TOKEN.sub(lambda m: self.word(m.group()), s)


def mask_example(example: Dict, fields: Sequence[str] = (
        "s1_name", "s1_addr", "cand_name", "cand_addr")) -> Dict:
    """Mask one error-analysis example in place-ish (returns a new dict)."""
    m = Masker()
    out = dict(example)
    for f in fields:
        if f in out and isinstance(out[f], str):
            out[f] = m.text(out[f])
    return out


def mask_examples(examples: Iterable[Dict]) -> List[Dict]:
    return [mask_example(e) for e in examples]
