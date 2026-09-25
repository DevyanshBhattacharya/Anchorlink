"""Module 2 — normalisation, views and per-country corpus statistics.

Three layers, in order of how much they assume:

1. Universal rules.  NFKC, romanise every script and fold accents (anyascii, ISC
   licence — *not* unidecode, which is GPL), lowercase, strip placeholders,
   split DBA/pipe aliases, reduce a domain to its label, strip leading zeros,
   expand number ranges.  Nothing here is language-specific, so it transfers to
   a country that never appears in training.
2. Corpus statistics per country, computed from unlabeled records: IDF, and an
   edge-position "affixness" statistic that recovers legal forms (SARL, SAS,
   EURL, LLC, Pvt Ltd, ...) without any word list.
3. Views per record: folded text, core name with affixes damped, the romanised
   consonant skeleton, the house-number set and the weighted address bag.
"""
from __future__ import annotations

import math
import re
import unicodedata
import zlib
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
from anyascii import anyascii  # ISC licence


def _stable_hash(s: str) -> int:
    """Deterministic across runs and machines, unlike Python's salted hash()."""
    return zlib.crc32(s.encode("utf-8"))


def _endless_none():
    while True:
        yield None

# --------------------------------------------------------------- regexes
PLACEHOLDER = re.compile(r"<\s*null\s*>|#{2,}|^[\W_]+|[\W_]+$", re.I)
#: A broad, deliberately country-neutral TLD list.  A bare ``[a-z]{2,6}`` suffix
#: would misfire on ordinary abbreviations ("pvt.ltd" -> label "pvt", TLD "ltd"),
#: so the set is enumerated; it spans generic and country-code TLDs well beyond
#: the three countries in this dataset, because a domain-style name is a noise
#: pattern, not a country signal.
_TLDS = ("com|net|org|co|io|biz|info|me|online|site|shop|store"
         "|in|fr|us|uk|de|es|it|nl|ca|au|jp|cn|br|ae|sg|za|ru|se|ch|be|pl|mx|id")
DOMAIN = re.compile(
    r"\b(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]{2,})\.(?:" + _TLDS + r")\b"
)
RANGE = re.compile(r"(\d+)\s*[-/]\s*(\d+)")
ALIAS_SPLIT = re.compile(r"\s+\|\s+|\s+(?:dba|d/b/a|t/a)\s+")
WORD = re.compile(r"[a-z0-9]+")
#: Anything outside Basic Latin through Latin Extended-B (U+0000-U+024F),
#: i.e. a genuinely non-Latin script rather than an accented Latin letter.
NON_LATIN = re.compile("[^\\u0000-\\u024f]")

@lru_cache(maxsize=200_000)
def _fold_uncached(s: str) -> str:
    return anyascii(unicodedata.normalize("NFKC", s)).lower()


# --------------------------------------------------------------- layer 1
def fold(s: str) -> str:
    """Script-, accent- and case-free view.

    LRU-cached: the same strings recur inside a partition, but a bulk pass over
    12M distinct records must not grow an unbounded dict.
    """
    return _fold_uncached(s) if s else ""


def clear_caches() -> None:
    _fold_uncached.cache_clear()


def has_native_script(s: str) -> bool:
    """True when the raw string carries characters outside Latin/Latin-1."""
    return bool(s) and bool(NON_LATIN.search(s))


def tokens(s: str) -> List[str]:
    """Alphanumeric tokens of the folded string, with immediate repeats dropped.

    "Leela Leela Diaz" -> ["leela", "diaz"] (a duplication pattern in the data).
    """
    t = WORD.findall(fold(s))
    return [w for i, w in enumerate(t) if i == 0 or w != t[i - 1]]


def clean_text(s: str) -> str:
    """Folded text with placeholder junk removed, for indexing."""
    s = fold(s)
    s = re.sub(r"<\s*null\s*>|#{2,}", " ", s)
    s = re.sub(r"\bnull\b", " ", s)
    return " ".join(WORD.findall(s))


def name_aliases(name: str) -> List[str]:
    """DBA parts, pipe-separated parts and bare domain labels, folded."""
    s = fold(name)
    s = re.sub(r"<\s*null\s*>|#{2,}", " ", s)
    s = re.sub(r"[\[\](){}]", " ", s)
    s = re.sub(r"^[\W_]+", " ", s)
    out: List[str] = []
    for part in ALIAS_SPLIT.split(s):
        m = DOMAIN.search(part)
        if m and len(part.split()) == 1:
            out.append(m.group(1).replace("-", " "))
        else:
            stripped = DOMAIN.sub(" ", part)
            out.append(stripped)
            if m:
                out.append(m.group(1).replace("-", " "))
    seen, res = set(), []
    for p in out:
        joined = " ".join(tokens(p))
        if joined and joined not in seen:
            seen.add(joined)
            res.append(joined)
    return res


def is_domain_name(name: str) -> bool:
    return bool(DOMAIN.search(fold(name)))


def house_numbers(addr: str) -> Set[str]:
    """Zero-stripped numeric tokens, with ranges expanded.

    ``003182`` -> ``3182``;  ``4109-4111`` -> ``{4109, 4111}``.
    """
    if not addr:
        return set()
    s = fold(addr)
    nums = {n.lstrip("0") or "0" for n in re.findall(r"\d+", s)}
    for a, b in RANGE.findall(s):
        nums |= {a.lstrip("0") or "0", b.lstrip("0") or "0"}
    return nums


def is_abbrev(a: str, b: str) -> bool:
    """Soft abbreviation test: rd~road, pvt~private, bd~boulevard, ste~societe.

    Same first letter, a subsequence, and at most 70% of the length.  It also
    fires on rd~reed, so callers must treat it as a partial similarity (~0.8),
    never as equality.
    """
    if not a or not b or len(a) >= len(b) or a[0] != b[0] or len(a) > 0.7 * len(b):
        return False
    it = iter(b)
    return all(ch in it for ch in a)


SKEL_RULES: Tuple[Tuple[str, str], ...] = (
    ("ph", "f"), ("sh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"),
    ("th", "t"), ("dh", "d"), ("bh", "b"),
    ("q", "k"), ("c", "k"), ("x", "ks"), ("w", "v"), ("z", "s"), ("y", "i"),
)
_VOWELS = re.compile(r"[aeiou]")
_NONSKEL = re.compile(r"[^a-z0-9 ]")
_RUNS = re.compile(r"(.)\1+")


def skeleton(s: str) -> str:
    """Romanised consonant skeleton — the cross-script blocking key.

    "Dynamic Hospitality Private Limited" and the Devanagari spelling of the same
    name both become ``dnmk hsptlt prvt lmtd``.  Median 3-gram Jaccard on
    native-script true pairs rises from 0.19 to 0.54; random pairs reach 0.41 at
    the 99th percentile, so it buys recall, never a decision.
    """
    s = fold(s)
    for a, b in SKEL_RULES:
        s = s.replace(a, b)
    s = _NONSKEL.sub(" ", _VOWELS.sub("", s))
    s = _RUNS.sub(r"\1", s)
    return " ".join(s.split())


def char_ngrams(s: str, n: int = 3) -> Set[str]:
    if not s:
        return set()
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


# --------------------------------------------------------------- layer 2
class TokenRoles:
    """Per-country statistics from unlabeled records: IDF + edge-position affixes.

    ``affix`` holds tokens that sit at the first or last position of a 3+-token
    name in at least ``edge_ratio`` of their occurrences and appear in at least
    ``min_share`` of such names.  On the test files this recovers sarl/sas/eurl/
    sasu/sci for France, limited/ltd/llp for India and llc/inc/corp for the US,
    with no dictionary.  Flagged tokens are damped (0.1x IDF), never deleted: a
    city inside a French trade name carries little information, but not none.
    """

    NAME_HASH_SLOTS = 1 << 24          # 16.8M slots, uint16 -> 33 MB per country

    def __init__(self, min_share: float = 0.003, edge_ratio: float = 0.85,
                 damp: float = 0.1, bigram_min_share: float = 0.002,
                 name_hash_slots: int | None = None):
        self.min_share = min_share
        self.edge_ratio = edge_ratio
        self.damp = damp
        self.bigram_min_share = bigram_min_share
        self.name_hash_slots = name_hash_slots or self.NAME_HASH_SLOTS
        self._acc: dict | None = None
        self.name_hash: Dict[str, "np.ndarray"] = {}
        self.idf: Dict[str, Dict[str, float]] = {}
        self.max_idf: Dict[str, float] = {}
        self.affix: Dict[str, Set[str]] = {}
        self.affix_bigram: Dict[str, Set[Tuple[str, str]]] = {}
        self.n_docs: Dict[str, int] = {}
        self.addr_idf: Dict[str, Dict[str, float]] = {}
        self.addr_max_idf: Dict[str, float] = {}

    # -- fitting -----------------------------------------------------------
    def _new_acc(self) -> dict:
        return {
            "df": defaultdict(Counter), "edge": defaultdict(Counter),
            "edge_bi": defaultdict(Counter), "bi_df": defaultdict(Counter),
            "n3": Counter(), "nd": Counter(),
            "adf": defaultdict(Counter), "nad": Counter(),
            "namefreq": {},
        }

    def partial_fit(self, names: Iterable[str], countries: Iterable[str],
                    addresses: Iterable[str] | None = None) -> "TokenRoles":
        """Accumulate statistics from one chunk.  Call :meth:`finalize` after."""
        if self._acc is None:
            self._acc = self._new_acc()
        a = self._acc
        df, edge, edge_bi, bi_df = a["df"], a["edge"], a["edge_bi"], a["bi_df"]
        n3, nd, adf, nad, namefreq = a["n3"], a["nd"], a["adf"], a["nad"], a["namefreq"]
        addr_iter = addresses if addresses is not None else _endless_none()
        for name, c, addr in zip(names, countries, addr_iter):
            t = tokens(name)
            nd[c] += 1
            df[c].update(set(t))
            if c not in namefreq:
                namefreq[c] = np.zeros(self.name_hash_slots, dtype=np.uint16)
            h = _stable_hash(" ".join(t)) % self.name_hash_slots
            if namefreq[c][h] < 65535:
                namefreq[c][h] += 1
            if len(t) >= 3:
                n3[c] += 1
                edge[c][t[0]] += 1
                edge[c][t[-1]] += 1
                edge_bi[c][(t[0], t[1])] += 1
                edge_bi[c][(t[-2], t[-1])] += 1
            for bg in zip(t, t[1:]):
                bi_df[c][bg] += 1
            if addr is not None:
                at = tokens(addr)
                if at:
                    nad[c] += 1
                    adf[c].update(set(at))
        return self

    def finalize(self, min_df: int = 2, min_addr_df: int = 2) -> "TokenRoles":
        """Compute IDF, affixes and prune the hapax tail.

        A token seen once has ``idf = log(N)``, which is exactly the fallback
        :meth:`weight` uses for an unseen token, so dropping it is lossless and
        removes ~60% of the vocabulary.
        """
        a = self._acc
        if a is None:
            raise RuntimeError("partial_fit was never called")
        df, edge, edge_bi, bi_df = a["df"], a["edge"], a["edge_bi"], a["bi_df"]
        n3, nd, adf, nad = a["n3"], a["nd"], a["adf"], a["nad"]

        self.n_docs = dict(nd)
        self.max_idf = {c: math.log(max(nd[c], 2)) for c in df}
        self.idf = {c: {w: math.log(nd[c] / d) for w, d in df[c].items() if d >= min_df}
                    for c in df}
        self.affix = {
            c: {w for w, e in edge[c].items()
                if df[c][w] >= self.min_share * max(n3[c], 1)
                and e / df[c][w] >= self.edge_ratio}
            for c in df
        }
        self.affix_bigram = {
            c: {bg for bg, e in edge_bi[c].items()
                if bi_df[c][bg] >= self.bigram_min_share * max(n3[c], 1)
                and e / bi_df[c][bg] >= self.edge_ratio}
            for c in edge_bi
        }
        if nad:
            self.addr_idf = {c: {w: math.log(nad[c] / d) for w, d in adf[c].items()
                                 if d >= min_addr_df}
                             for c in adf}
            self.addr_max_idf = {c: math.log(max(nad[c], 2)) for c in adf}
        self.name_hash = a["namefreq"]
        self._acc = None
        return self

    def fit(self, names: Iterable[str], countries: Iterable[str],
            addresses: Iterable[str] | None = None,
            min_df: int = 1, min_addr_df: int = 1) -> "TokenRoles":
        """One-shot fit (small inputs and tests)."""
        return self.partial_fit(names, countries, addresses).finalize(
            min_df=min_df, min_addr_df=min_addr_df)

    # -- use ---------------------------------------------------------------
    def weight(self, w: str, c: str) -> float:
        """Damped IDF for a name token.  Unseen token -> the country's max IDF."""
        idf = self.idf.get(c, {}).get(w)
        if idf is None:
            idf = self.max_idf.get(c, 10.0)
        return self.damp * idf if w in self.affix.get(c, ()) else idf

    def addr_weight(self, w: str, c: str) -> float:
        """Damped IDF for an address token; falls back to the name statistics."""
        table = self.addr_idf.get(c)
        if table is None:
            return self.weight(w, c)
        idf = table.get(w)
        if idf is None:
            idf = self.addr_max_idf.get(c, 10.0)
        return idf

    def is_affix(self, w: str, c: str) -> bool:
        return w in self.affix.get(c, ())

    def max_name_idf(self, c: str) -> float:
        """The country's ceiling on name IDF — what an unseen token is worth.

        Rarity features divide by it so "this pair agreed on a rare word" means
        the same thing in a country whose corpus is a different size, which is
        the only way such a feature survives leave-one-country-out.
        """
        return float(self.max_idf.get(c, 10.0))

    def max_addr_idf(self, c: str) -> float:
        return float(self.addr_max_idf.get(c, self.max_idf.get(c, 10.0)))

    def core_tokens(self, name: str, c: str) -> List[str]:
        """Name tokens with flagged affixes (and affix bigrams) removed.

        Stripping is backed off in three steps rather than applied blindly, so a
        name made entirely of frequent edge tokens ("Private Limited") still has
        a core instead of collapsing to nothing.
        """
        t = tokens(name)
        if not t:
            return t
        affix = self.affix.get(c, ())
        bigr = self.affix_bigram.get(c, ())
        keep = [True] * len(t)
        if len(t) >= 3:
            for i in (0, len(t) - 2):
                bg = (t[i], t[i + 1])
                if bg in bigr:
                    keep[i] = keep[i + 1] = False
        core = [w for w, k in zip(t, keep) if k and w not in affix]
        if core:
            return core
        core = [w for w in t if w not in affix]      # back off: bigrams only
        if core:
            return core
        core = [w for w, k in zip(t, keep) if k]     # back off: affixes only
        return core or t

    def core_name(self, name: str, c: str) -> str:
        return " ".join(self.core_tokens(name, c))

    def name_freq_pct(self, name: str, c: str) -> float:
        """How common this exact normalised name is in the partition, in [0,1].

        Backed by a 2**24-slot uint16 hash counter per country (33 MB) rather
        than a name -> count dict, which would cost gigabytes at 12M records.
        Hash collisions only inflate a rare name's count slightly, which is
        harmless for a log-scaled rarity feature.
        """
        table = self.name_hash.get(c)
        if table is None:
            return float("nan")
        h = _stable_hash(" ".join(tokens(name))) % len(table)
        n = int(table[h])
        total = self.n_docs.get(c, 1)
        return math.log1p(n) / math.log1p(max(total, 2))

    def subset(self, country: str) -> "TokenRoles":
        """A copy holding one country's statistics only.

        Feature workers see a single partition, and the full object carries a
        33 MB name-frequency table per country plus that country's IDF tables.
        Handing eight workers the whole thing wastes hundreds of megabytes.
        """
        obj = TokenRoles(self.min_share, self.edge_ratio, self.damp,
                         self.bigram_min_share, self.name_hash_slots)
        for attr in ("idf", "max_idf", "affix", "affix_bigram", "n_docs",
                     "addr_idf", "addr_max_idf", "name_hash"):
            src = getattr(self, attr, None) or {}
            if country in src:
                setattr(obj, attr, {country: src[country]})
        return obj

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "min_share": self.min_share, "edge_ratio": self.edge_ratio,
            "damp": self.damp, "bigram_min_share": self.bigram_min_share,
            "idf": {c: dict(v) for c, v in self.idf.items()},
            "max_idf": self.max_idf,
            "affix": {c: sorted(v) for c, v in self.affix.items()},
            "affix_bigram": {c: sorted(list(x) for x in v)
                             for c, v in self.affix_bigram.items()},
            "n_docs": self.n_docs,
            "addr_idf": {c: dict(v) for c, v in self.addr_idf.items()},
            "addr_max_idf": self.addr_max_idf,
            "name_hash_slots": self.name_hash_slots,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TokenRoles":
        obj = cls(d.get("min_share", 0.003), d.get("edge_ratio", 0.85),
                  d.get("damp", 0.1), d.get("bigram_min_share", 0.002))
        obj.idf = d["idf"]
        obj.max_idf = d["max_idf"]
        obj.affix = {c: set(v) for c, v in d["affix"].items()}
        obj.affix_bigram = {c: {tuple(x) for x in v}
                            for c, v in d.get("affix_bigram", {}).items()}
        obj.n_docs = d["n_docs"]
        obj.addr_idf = d.get("addr_idf", {})
        obj.addr_max_idf = d.get("addr_max_idf", {})
        obj.name_hash_slots = d.get("name_hash_slots", cls.NAME_HASH_SLOTS)
        return obj


# --------------------------------------------------------------- views
def index_view(name: str, addr: str) -> str:
    """The text the sparse retriever indexes: folded name, then folded address."""
    return f"{clean_text(name)} | {clean_text(addr)}"


def skeleton_view(name: str, addr: str) -> str:
    """The cross-script retrieval view."""
    return f"{skeleton(name)} | {skeleton(addr)}"
