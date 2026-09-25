"""Generate a miniature challenge dataset with the same shape as the real one.

Three countries, one of which (``Carpathia``) exists only in the test split, so
the end-to-end run exercises the unseen-country path.  Noise operators are the
ones observed in the real files: zero padding, number truncation, legal-form
moves, abbreviations, token duplication, placeholders, character corruption,
domain-style names and a non-Latin script.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

BRANDS = ["apex", "orion", "harbour", "zenith", "kestrel", "lumen", "vertex", "quill",
          "solace", "ember", "onyx", "aurora", "cobalt", "marlin", "juniper", "sable",
          "thistle", "pelican", "quarry", "lantern"]
TYPES = ["digital", "foods", "textiles", "logistics", "hospitality", "metals",
         "finance", "pharma", "traders", "engineering"]
STREETS = ["westchester drive", "ellis road", "montebello avenue", "orville avenue",
           "lynchburg street", "dolley madison blvd", "mg road", "gandhi marg",
           "rue de dieppe", "boulevard roosevelt"]
CITIES = {"Atlantis": ["high point", "tahlequah", "phoenix", "columbus"],
          "Borduria": ["jaipur", "kochi", "bhopal", "gurgaon"],
          "Carpathia": ["bordeaux", "lille", "dunkerque", "nantes"]}
SUFFIX = {"Atlantis": ["llc", "inc", "corp"],
          "Borduria": ["private limited", "ltd", "llp"],
          "Carpathia": ["sarl", "sas", "eurl"]}
ABBREV = {"drive": "dr", "road": "rd", "avenue": "ave", "street": "st",
          "boulevard": "blvd", "private limited": "pvt ltd", "limited": "ltd"}
# a fake script, to exercise the romanise + skeleton path without shipping real text
SCRIPT = str.maketrans("abcdefghijklmnopqrstuvwxyz",
                       "абсдефґһіј"
                       "кӁмпорҡгѕт"
                       "цвшхуз")


def noisy_name(name: str, rng: random.Random) -> str:
    r = rng.random()
    toks = name.split()
    if r < 0.12 and len(toks) > 2:                       # legal-form move
        name = " ".join([toks[-1]] + toks[:-1])
    elif r < 0.22:                                       # abbreviation
        for long, short in ABBREV.items():
            name = name.replace(long, short)
    elif r < 0.30 and len(toks) > 1:                     # token duplication
        name = " ".join([toks[0], toks[0]] + toks[1:])
    elif r < 0.38:                                       # character corruption
        i = rng.randrange(len(name))
        name = name[:i] + rng.choice("aeiou1") + name[i + 1:]
    elif r < 0.44:                                       # domain-style name
        name = "".join(toks[:2]) + ".com"
    elif r < 0.52:                                       # non-Latin script
        name = name.translate(SCRIPT)
    elif r < 0.56:                                       # placeholder junk
        name = "-- " + name
    return name.upper() if rng.random() < 0.3 else name


def noisy_addr(num: int, street: str, city: str, rng: random.Random) -> str:
    r = rng.random()
    n = str(num)
    if r < 0.15:
        n = n.zfill(6)                                   # zero padding
    elif r < 0.22 and len(n) > 3:
        n = n[1:]                                        # truncation
    elif r < 0.27:
        n = f"{num}-{num + 2}"                           # range
    s = street
    if rng.random() < 0.35:
        for long, short in ABBREV.items():
            s = s.replace(long, short)
    parts = [f"{n} {s}", city]
    if rng.random() < 0.15:
        parts.append("near the old mill")                # landmark
    if rng.random() < 0.2:
        parts.reverse()                                  # component reordering
    if rng.random() < 0.04:
        return ""                                        # missing address
    return ", ".join(parts)


def generate(out: Path, countries: List[str], n_s1: int, seed: int,
             split: str, with_gt: bool) -> None:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    s1_rows, s2_rows, s3_rows, gt_rows = [], [], [], []
    uid = seed * 10_000_000

    def nid(prefix: str) -> str:
        nonlocal uid
        uid += 1
        return f"{prefix}-{uid}"

    for country in countries:
        for _ in range(n_s1):
            brand = rng.choice(BRANDS)
            typ = rng.choice(TYPES)
            suf = rng.choice(SUFFIX[country])
            name = f"{brand} {typ} {suf}"
            num = rng.randrange(10, 9999)
            street = rng.choice(STREETS)
            city = rng.choice(CITIES[country])
            s1_id = nid("S1")
            s1_rows.append((s1_id, name.title(), f"{num} {street.title()}, {city.title()}",
                            country))
            matches: List[str] = []
            n_match = rng.choices([0, 1, 2, 3, 4], weights=[6, 12, 34, 30, 18])[0]
            for j in range(n_match):
                src = "S2" if j % 2 == 0 else "S3"
                rid = nid(src)
                row = (rid, noisy_name(name, rng), noisy_addr(num, street, city, rng),
                       country)
                (s2_rows if src == "S2" else s3_rows).append(row)
                matches.append(rid)
            gt_rows.append((s1_id, ",".join(matches)))
            # a distractor: same name, a few doors down
            if rng.random() < 0.6:
                src = rng.choice(["S2", "S3"])
                rid = nid(src)
                row = (rid, noisy_name(name, rng),
                       noisy_addr(num + rng.randrange(3, 20), street, city, rng), country)
                (s2_rows if src == "S2" else s3_rows).append(row)

    def write(path: Path, rows, header):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(header) + "\n")
            for r in rows:
                f.write("\t".join(x.replace("\t", " ") for x in r) + "\n")

    hdr = ["entity_id", "business_name", "business_address", "country"]
    write(out / f"{split}_source1.tsv", s1_rows, hdr)
    write(out / f"{split}_source2.tsv", s2_rows, hdr)
    write(out / f"{split}_source3.tsv", s3_rows, hdr)
    if with_gt:
        write(out / f"{split}_ground_truth.tsv", gt_rows,
              ["source1_entity_id", "matched_entity_ids"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    generate(a.root / "train", ["Atlantis", "Borduria"], a.n, a.seed, "train", True)
    generate(a.root / "test", ["Atlantis", "Borduria", "Carpathia"],
             max(a.n // 2, 200), a.seed + 1, "test", False)
    print(f"wrote synthetic dataset to {a.root}")


if __name__ == "__main__":
    main()
