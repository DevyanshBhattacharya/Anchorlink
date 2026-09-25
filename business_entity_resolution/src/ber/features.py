"""Module 4.4 — explicit pair features.

Every feature is a ratio, a percentile or a class index, never a raw corpus
count, so it means the same thing in a country the model never saw.  Country
itself is never a feature: it is the partition key and nothing else.

Two design rules carried from the research doc:

* **Never ask a model to do house-number arithmetic.**  85.9% of true pairs share
  a number against 12.7% of same-name lookalikes, whose commonest pattern is a
  neighbouring number on the same street.  Numbers get their own relation class
  and a graded distance.
* **Compare addresses by containment, not Jaccard.**  One side routinely carries
  extra landmarks, administrative units or PO boxes ("Near Fortis Hospital"), so
  unmatched weight is reported per side and left for the model to weigh.

Missing fields are never imputed: they become NaN plus a flag, and LightGBM
routes NaN natively.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .normalize import (TokenRoles, char_ngrams, clean_text, fold, has_native_script,
                        house_numbers, is_abbrev, is_domain_name, name_aliases,
                        skeleton, tokens)

NAN = float("nan")


# ------------------------------------------------------------------ views
class RecordView:
    """Everything about one record that a pair feature could need, computed once."""
    __slots__ = ("name", "addr", "core", "core_toks", "name_toks", "addr_toks",
                 "addr_w", "addr_W", "core_w", "core_W", "name_grams", "skel",
                 "skel_grams", "addr_grams", "nums", "nums_i", "aliases",
                 "native", "domain", "freq_pct", "has_addr")

    def __init__(self, name: str, addr: str, country: str, roles: TokenRoles):
        self.name = clean_text(name)
        self.addr = clean_text(addr)
        self.name_toks = tokens(name)
        self.core_toks = roles.core_tokens(name, country) or self.name_toks
        self.core = " ".join(self.core_toks)
        self.addr_toks = tokens(addr)
        self.addr_w = {t: roles.addr_weight(t, country) for t in set(self.addr_toks)}
        self.addr_W = float(sum(self.addr_w.values()))
        self.core_w = {t: roles.weight(t, country) for t in set(self.core_toks)}
        self.core_W = float(sum(self.core_w.values()))
        self.name_grams = char_ngrams(self.core.replace(" ", ""))
        # The skeleton is built from the **full** name, not the core.  Its job is
        # to bridge scripts, and affix stripping is asymmetric across a
        # transliteration: "Private Limited" is recognised and stripped, while
        # its romanised form "praivet limited" is not, so a core-based skeleton
        # compares three tokens against two and the bridge it was built for
        # stops working.
        self.skel = skeleton(self.name).replace(" ", "")
        self.skel_grams = char_ngrams(self.skel)
        self.addr_grams = char_ngrams(self.addr.replace(" ", ""))
        self.nums = house_numbers(addr)
        self.nums_i = sorted(int(n[:9]) for n in self.nums) if self.nums else []
        self.aliases = name_aliases(name)
        self.native = has_native_script(name) or has_native_script(addr)
        self.domain = is_domain_name(name)
        self.freq_pct = roles.name_freq_pct(name, country)
        self.has_addr = bool(self.addr_toks)


def build_views(names: Sequence[str], addrs: Sequence[str], country: str,
                roles: TokenRoles) -> List[RecordView]:
    return [RecordView(n, a, country, roles) for n, a in zip(names, addrs)]


# ------------------------------------------------------------------ helpers
def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return NAN
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _tok_sim(x: str, y: str) -> float:
    """Inner similarity for Monge–Elkan: equality, soft abbreviation, then JW.

    The abbreviation test fires on rd~road and bd~boulevard but also on rd~reed,
    so it contributes ~0.8, never 1.0.
    """
    if x == y:
        return 1.0
    if is_abbrev(x, y) or is_abbrev(y, x):
        return 0.8
    return JaroWinkler.normalized_similarity(x, y)


def monge_elkan(a: Sequence[str], b: Sequence[str], w: Dict[str, float],
                default_w: float = 1.0) -> float:
    if not a or not b:
        return NAN
    num = den = 0.0
    for x in a:
        wx = w.get(x, default_w)
        num += wx * max(_tok_sim(x, y) for y in b)
        den += wx
    return num / den if den else NAN


def soft_tfidf(a: Sequence[str], wa: Dict[str, float],
               b: Sequence[str], wb: Dict[str, float], theta: float = 0.90) -> float:
    """Cohen, Ravikumar & Fienberg (2003): rare-token agreement under fuzziness."""
    if not a or not b:
        return NAN
    na = math.sqrt(sum(wa.get(t, 1.0) ** 2 for t in set(a)))
    nb = math.sqrt(sum(wb.get(t, 1.0) ** 2 for t in set(b)))
    if na == 0 or nb == 0:
        return NAN
    total = 0.0
    for x in set(a):
        best, best_y = 0.0, None
        for y in set(b):
            s = _tok_sim(x, y)
            if s > best:
                best, best_y = s, y
        if best >= theta and best_y is not None:
            total += wa.get(x, 1.0) * wb.get(best_y, 1.0) * best
    return total / (na * nb)


def _one_sided(a: Set[str], b: Set[str], affixes: Set[str]) -> int:
    """Non-affix tokens of ``a`` with no counterpart in ``b``.

    A counterpart is an identical token or an abbreviation of one in either
    direction, so ``ltd``/``limited`` does not read as a missing word.  Flagged
    legal forms are excluded from the count because dropping one is ordinary
    noise; dropping an ordinary word is not.
    """
    n = 0
    for t in a:
        if t in b or t in affixes:
            continue
        if any(is_abbrev(t, o) or is_abbrev(o, t) for o in b):
            continue
        n += 1
    return n


def number_relation(n1: Set[str], n2: Set[str]) -> int:
    """The classes measured in finding 5 of the research doc.

    0 missing · 1 a number in common · 2 prefix/suffix truncation (1447 -> 447)
    3 differ by 1-2 · 4 differ by 3-20 · 5 other
    """
    if not n1 or not n2:
        return 0
    if n1 & n2:
        return 1
    for x in n1:
        for y in n2:
            if len(x) > 1 and len(y) > 1 and (x.endswith(y) or y.endswith(x)
                                              or x.startswith(y) or y.startswith(x)):
                return 2
    d = min(abs(int(x[:9]) - int(y[:9])) for x in n1 for y in n2)
    return 3 if d <= 2 else 4 if d <= 20 else 5


# ------------------------------------------------------------------ features
PAIR_FEATURES: Tuple[str, ...] = (
    # name
    "name_jw", "name_lev", "name_tsort", "name_tset",
    "name_gram_jac", "name_skel_jac",
    "name_me_12", "name_me_21", "name_softtfidf",
    "name_idf_contain", "name_idf_unmatched_1", "name_idf_unmatched_2",
    "name_freq_max", "alias_best", "name_len_ratio",
    "affix_rel", "name_tok_jac", "name_tok_only_1", "name_tok_only_2", "name_tok_eq",
    "domain_flag", "native_flag",
    # address
    "addr_missing", "addr_contain", "addr_unmatched_1", "addr_unmatched_2",
    "addr_gram_jac", "addr_me", "addr_len_ratio", "locality_conflict",
    # numbers
    "num_rel", "num_min_diff", "num_shared", "num_unmatched_1", "num_unmatched_2",
    # retrieval (one score/rank pair per retriever, then the fusion columns)
    "r_na_score", "r_na_rank", "r_nm_score", "r_nm_rank",
    "r_ad_score", "r_ad_rank", "r_sk_score", "r_sk_rank",
    "r_key_hit", "r_rrf", "r_nret", "r_pre",
    # set-level, computable straight from the candidate list
    "x_support_name", "x_support_addr", "x_rrf_rank", "x_rrf_gap", "x_n_cands",
)
N_PAIR = len(PAIR_FEATURES)
N_PAIRWISE_ONLY = 35          # columns pair_feature_row fills
I_RETRIEVAL = 35              # retrieval columns come from the candidate table
I_SET = 47                    # set columns come from the candidate list

#: Bumped whenever the feature set changes, so cached shards from an older
#: definition are never silently mixed with new ones.
FEATURE_VERSION = 2

#: Added by the meta stage once stage-1 probabilities exist.
META_FEATURES: Tuple[str, ...] = (
    "m_prob", "m_rank", "m_gap_to_best", "m_score_share", "m_n_strong",
    "m_excl_prob", "m_excl_rank", "m_n_competitors", "m_best_other_source",
)

DEEP_FEATURES: Tuple[str, ...] = ("ce_logit", "llm_logit")

# +1 increasing, -1 decreasing, 0 unconstrained.  Monotone constraints remove
# non-monotone splits that would not transfer to an unseen country.
_MONO: Dict[str, int] = {
    "name_jw": 1, "name_lev": 1, "name_tsort": 1, "name_tset": 1,
    "name_gram_jac": 1, "name_skel_jac": 1, "name_me_12": 1, "name_me_21": 1,
    "name_softtfidf": 1, "name_idf_contain": 1,
    "name_idf_unmatched_1": -1, "name_idf_unmatched_2": -1,
    "alias_best": 1,
    "name_tok_jac": 1, "name_tok_only_1": -1, "name_tok_only_2": -1, "name_tok_eq": 1,
    "addr_contain": 1, "addr_unmatched_1": -1, "addr_unmatched_2": -1,
    "addr_gram_jac": 1, "addr_me": 1, "locality_conflict": -1,
    "num_shared": 1, "num_min_diff": -1,
    "r_na_score": 1, "r_nm_score": 1, "r_ad_score": 1, "r_sk_score": 1,
    "r_na_rank": -1, "r_nm_rank": -1, "r_ad_rank": -1, "r_sk_rank": -1,
    "r_rrf": 1, "r_nret": 1, "r_pre": 1,
    "x_support_name": 1, "x_support_addr": 1, "x_rrf_gap": -1,
    "m_prob": 1, "m_excl_prob": 1, "m_gap_to_best": -1, "m_best_other_source": 0,
    "ce_logit": 1, "llm_logit": 1,
}


def monotone_vector(feature_names: Sequence[str]) -> List[int]:
    return [_MONO.get(f, 0) for f in feature_names]


def pair_feature_row(v1: RecordView, v2: RecordView, roles: TokenRoles,
                     country: str, out: np.ndarray) -> None:
    """Fill ``out`` (length ``N_PAIR``) with the pairwise features for one pair."""
    c1, c2 = v1.core or v1.name, v2.core or v2.name
    out[0] = JaroWinkler.normalized_similarity(c1, c2)
    out[1] = Levenshtein.normalized_similarity(c1, c2)
    out[2] = fuzz.token_sort_ratio(c1, c2) / 100.0
    out[3] = fuzz.token_set_ratio(c1, c2) / 100.0
    out[4] = jaccard(v1.name_grams, v2.name_grams)
    out[5] = jaccard(v1.skel_grams, v2.skel_grams)
    out[6] = monge_elkan(v1.core_toks, v2.core_toks, v1.core_w)
    out[7] = monge_elkan(v2.core_toks, v1.core_toks, v2.core_w)
    out[8] = soft_tfidf(v1.core_toks, v1.core_w, v2.core_toks, v2.core_w)

    shared = set(v1.core_w) & set(v2.core_w)
    inter_w = sum(v1.core_w[t] for t in shared)
    if v1.core_W > 0 and v2.core_W > 0:
        out[9] = inter_w / min(v1.core_W, v2.core_W)
        out[10] = (v1.core_W - inter_w) / v1.core_W
        out[11] = (v2.core_W - sum(v2.core_w[t] for t in shared)) / v2.core_W
    else:
        out[9] = out[10] = out[11] = NAN

    f1, f2 = v1.freq_pct, v2.freq_pct
    out[12] = max(f1, f2) if (f1 == f1 and f2 == f2) else (f1 if f1 == f1 else f2)
    out[13] = (max(fuzz.token_set_ratio(a, b) for a in v1.aliases for b in v2.aliases) / 100.0
               if v1.aliases and v2.aliases else NAN)
    l1, l2 = len(c1), len(c2)
    out[14] = min(l1, l2) / max(l1, l2, 1)

    # -- words one side has and the other does not, counted, not weighted
    #
    # This is the feature the error analysis demanded, and it has to look at the
    # **full** token list, not the core name.  Worked example from the training
    # data: S1-499359562 "Valiant Entertainment Private Limited" has three true
    # matches, all keeping *Private* — and "Valiant Entertainment Limited", at a
    # byte-identical address, matches no S1 at all.  Every other feature is blind
    # to that word:
    #   * the core name strips it, because the affix bigram (private, limited)
    #     is flagged, so both records reduce to "valiant entertainment";
    #   * `token_set_ratio` returns a flat 100 whenever one set contains the other;
    #   * `private` is common in Indian names, so its IDF is low and the weighted
    #     unmatched-mass features barely move.
    # Counting *non-affix* one-sided tokens separates them cleanly: the distractor
    # drops `private` (not a flagged affix, so it counts), while the true match
    # differs only by `limited` vs `ltd` (both affixes, and abbreviations of each
    # other, so neither counts).
    t1, t2 = set(v1.name_toks), set(v2.name_toks)
    af1 = {t for t in t1 if roles.is_affix(t, country)}
    af2 = {t for t in t2 if roles.is_affix(t, country)}
    out[15] = 0.0 if not af1 and not af2 else (1.0 if not af1 or not af2
                                               else (2.0 if af1 == af2 else 3.0))
    union_t = t1 | t2
    out[16] = len(t1 & t2) / len(union_t) if union_t else NAN
    out[17] = float(_one_sided(t1, t2, af1))
    out[18] = float(_one_sided(t2, t1, af2))
    out[19] = float(t1 == t2)
    out[20] = float(v1.domain or v2.domain)
    out[21] = float(v1.native) + float(v2.native)

    # -- address
    out[22] = float(not v1.has_addr) + float(not v2.has_addr)
    if v1.has_addr and v2.has_addr:
        sh = set(v1.addr_w) & set(v2.addr_w)
        iw1 = sum(v1.addr_w[t] for t in sh)
        iw2 = sum(v2.addr_w[t] for t in sh)
        denom = min(v1.addr_W, v2.addr_W)
        out[23] = iw1 / denom if denom > 0 else NAN
        out[24] = (v1.addr_W - iw1) / v1.addr_W if v1.addr_W > 0 else NAN
        out[25] = (v2.addr_W - iw2) / v2.addr_W if v2.addr_W > 0 else NAN
        out[26] = jaccard(v1.addr_grams, v2.addr_grams)
        alpha1 = [t for t in v1.addr_toks if not t.isdigit()]
        alpha2 = [t for t in v2.addr_toks if not t.isdigit()]
        out[27] = monge_elkan(alpha1, alpha2, v1.addr_w)
        out[28] = min(len(v1.addr_toks), len(v2.addr_toks)) / max(len(v1.addr_toks),
                                                                  len(v2.addr_toks), 1)
        # rare alphabetic tokens present on one side only (Jaipur vs Adilabad)
        rare1 = sum(w for t, w in v1.addr_w.items()
                    if t not in v2.addr_w and not t.isdigit() and w >= 6.0)
        rare2 = sum(w for t, w in v2.addr_w.items()
                    if t not in v1.addr_w and not t.isdigit() and w >= 6.0)
        tot = max(v1.addr_W, v2.addr_W)
        out[29] = (rare1 + rare2) / tot if tot > 0 else NAN
    else:
        out[23] = out[24] = out[25] = out[26] = out[27] = out[28] = out[29] = NAN

    # -- numbers
    n1, n2 = v1.nums, v2.nums
    out[30] = float(number_relation(n1, n2))
    if n1 and n2:
        d = min(abs(x - y) for x in v1.nums_i for y in v2.nums_i)
        out[31] = math.log1p(d)
        out[32] = float(len(n1 & n2))
        out[33] = len(n1 - n2) / len(n1)
        out[34] = len(n2 - n1) / len(n2)
    else:
        out[31] = out[32] = out[33] = out[34] = NAN


def support_similarity(v1: RecordView, v2: RecordView) -> Tuple[float, float]:
    """Cheap agreement between two *candidate* records (one S2, one S3).

    Agreement between an S2 and an S3 record is free evidence about both: if the
    S1's best S2 candidate and this S3 candidate describe the same place, the S3
    candidate is more likely to belong to the same cluster.  Only precomputed
    gram sets are touched, so this costs two set intersections.
    """
    n = jaccard(v1.name_grams, v2.name_grams)
    a = jaccard(v1.addr_grams, v2.addr_grams)
    return n, a


def group_starts(group_ids: np.ndarray) -> np.ndarray:
    """Start offsets of each run in a sorted group-id array, plus the end."""
    if len(group_ids) == 0:
        return np.zeros(1, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, group_ids[1:] != group_ids[:-1]])
    return np.r_[starts, len(group_ids)].astype(np.int64)


def rank_within_groups(bounds: np.ndarray, score: np.ndarray) -> np.ndarray:
    """0-based rank of every row inside its group, best score first."""
    out = np.empty(len(score), dtype=np.int32)
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = score[a:b]
        order = np.argsort(-seg, kind="stable")
        r = np.empty(b - a, dtype=np.int32)
        r[order] = np.arange(b - a, dtype=np.int32)
        out[a:b] = r
    return out


def best_within_groups(bounds: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Per-row copy of its group's maximum score."""
    out = np.empty(len(score), dtype=np.float64)
    for a, b in zip(bounds[:-1], bounds[1:]):
        out[a:b] = score[a:b].max() if b > a else 0.0
    return out


def sum_within_groups(bounds: np.ndarray, score: np.ndarray) -> np.ndarray:
    out = np.empty(len(score), dtype=np.float64)
    for a, b in zip(bounds[:-1], bounds[1:]):
        out[a:b] = score[a:b].sum() if b > a else 0.0
    return out


def best_other_source(bounds: np.ndarray, score: np.ndarray,
                      src: np.ndarray) -> np.ndarray:
    """Per row: the best score among this S1's candidates from the *other* source."""
    out = np.zeros(len(score), dtype=np.float64)
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg_s, seg_src = score[a:b], src[a:b]
        m2 = seg_s[seg_src == 2].max(initial=0.0)
        m3 = seg_s[seg_src == 3].max(initial=0.0)
        out[a:b] = np.where(seg_src == 2, m3, m2)
    return out


def build_meta_features(s1_codes: np.ndarray, cand_codes: np.ndarray,
                        src: np.ndarray, prob: np.ndarray,
                        strong: float = 0.5) -> np.ndarray:
    """The nine meta columns, from stage-1 probabilities alone.

    Rows must already be grouped by ``s1_codes`` (sorted).  ``cand_codes`` are
    integer ids of the candidate records, shared across S1 entities, so the
    exclusivity columns can see every claim on the same record.
    """
    n = len(prob)
    out = np.empty((n, len(META_FEATURES)), dtype=np.float32)
    bounds = group_starts(s1_codes)
    best = best_within_groups(bounds, prob)
    total = sum_within_groups(bounds, prob)
    out[:, 0] = prob
    out[:, 1] = rank_within_groups(bounds, prob)
    out[:, 2] = best - prob
    out[:, 3] = np.where(total > 0, prob / np.maximum(total, 1e-9), 0.0)
    strong_count = np.zeros(n, dtype=np.float32)
    for a, b in zip(bounds[:-1], bounds[1:]):
        strong_count[a:b] = float((prob[a:b] >= strong).sum())
    out[:, 4] = strong_count

    from .decide import exclusivity_arrays
    excl = exclusivity_arrays(s1_codes.astype(np.int64), cand_codes.astype(np.int64), prob)
    out[:, 5] = excl
    # rank of this S1's claim among all S1s claiming the same candidate
    order = np.lexsort((-prob, cand_codes))
    cb = group_starts(cand_codes[order])
    claim_rank = np.empty(n, dtype=np.float32)
    n_comp = np.empty(n, dtype=np.float32)
    for a, b in zip(cb[:-1], cb[1:]):
        claim_rank[order[a:b]] = np.arange(b - a, dtype=np.float32)
        n_comp[order[a:b]] = float(b - a)
    out[:, 6] = claim_rank
    out[:, 7] = n_comp
    out[:, 8] = best_other_source(bounds, prob, src)
    return out
