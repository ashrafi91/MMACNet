#!/usr/bin/env python3
"""Reconstruct the rare ICD-9 subset (Section 3.1 of the manuscript).

Pipeline
--------
1. Read the Orphanet nomenclature pack (release December 2025): every
   ORPHAcode -> ICD-10 alignment plus its relation qualifier
   (``E`` exact / ``NTBT`` narrower-to-broader / ``BTNT`` broader-to-narrower /
   ``ND`` undetermined).
2. Map each ICD-10 code to ICD-9 through the CMS ICD-10-CM/PCS General
   Equivalence Mappings (GEMs).  ICD-10 codes with no GEM match are discarded.
3. Write ``rare_disease_reference.csv`` -- the complete crosswalk with columns
   ``ORPHAcode, icd10_code, icd9_code, mapping_relation`` (Appendix B) -- and
   the ICD-9-only ``rare_icd9_candidate_codes.csv``.
4. If a MIMIC-III CSV directory is given, intersect the candidate ICD-9 codes
   with the codes that actually appear on admissions carrying a discharge
   summary and report the size of that set (the rare label space --
   ``EXPERIMENT_SPEC.yaml: dataset.rare.distinct_icd9_codes`` == 568).

The script is deterministic.  It needs the licensed Orphanet pack and the CMS
GEMs, which cannot be redistributed here; only the resulting ICD-9 candidate
list ships in ``supplementary/``.
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from MMACNet.modules.preprocessors import CodeProcessor

RELATIONS = {"E", "NTBT", "BTNT", "ND"}


def read_orphanet_pack(path: Path):
    """Yield ``(orphacode, icd10_code, relation)`` triples.

    Accepts either the Orphanet XML nomenclature pack or a pre-flattened CSV
    with columns ``ORPHAcode,ICD10,relation``.  Only the CSV path is
    implemented here; adapt ``_parse_xml`` for the raw pack.
    """
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rel = (row.get("relation") or row.get("mapping_relation") or "ND").strip().upper()
                yield (
                    str(row["ORPHAcode"]).strip(),
                    str(row.get("ICD10") or row.get("icd10_code")).strip(),
                    rel if rel in RELATIONS else "ND",
                )
    else:
        raise NotImplementedError(
            "Point --orphanet-nomenclature-pack at a flattened CSV "
            "(ORPHAcode,ICD10,relation) or extend _parse_xml()."
        )


def read_gem_crosswalk(path: Path):
    """Return ``{icd10_code: set(icd9_code)}`` from a CMS GEMs file.

    CMS GEM lines look like ``A0100  0020   00000``: source (ICD-10), target
    (ICD-9), then flags.  ``NoDx`` / all-zero targets are dropped.
    """
    mapping: dict[str, set[str]] = {}
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            src, tgt = parts[0].strip().upper(), parts[1].strip().upper()
            if tgt in {"NODX", "00000", "0000"} or set(tgt) == {"0"}:
                continue
            mapping.setdefault(src, set()).add(tgt)
    return mapping


def icd10_key(code: str) -> str:
    return code.replace(".", "").upper().strip()


def icd9_display(code: str, is_diagnosis: bool = True) -> str:
    """Re-insert the decimal point the way the preprocessing pipeline does."""
    return CodeProcessor.reformat_icd_code(code, is_diagnosis)


def build_reference(orphanet_pack: Path, gem_crosswalk: Path):
    gem = read_gem_crosswalk(gem_crosswalk)
    rows, seen = [], set()
    for orphacode, icd10, relation in read_orphanet_pack(orphanet_pack):
        targets = gem.get(icd10_key(icd10))
        if not targets:
            continue
        for icd9 in sorted(targets):
            icd9_disp = icd9_display(icd9)
            key = (orphacode, icd10, icd9_disp)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "ORPHAcode": orphacode,
                    "icd10_code": icd10,
                    "icd9_code": icd9_disp,
                    "mapping_relation": relation,
                }
            )
    rows.sort(key=lambda r: (r["ORPHAcode"], r["icd10_code"], r["icd9_code"]))
    return rows


def write_csv(rows, path: Path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_effective_codes(candidate_codes, mimic_csv_dir: Path) -> int:
    import pandas as pd

    def load(name, is_diag):
        df = pd.read_csv(
            mimic_csv_dir / name, dtype={"HADM_ID": "string", "ICD9_CODE": "string"},
            compression="gzip",
        )
        df["ICD9_CODE"] = df["ICD9_CODE"].apply(
            lambda c: CodeProcessor.reformat_icd_code(str(c), is_diag)
        )
        return df[["HADM_ID", "ICD9_CODE"]]

    codes = pd.concat(
        [load("DIAGNOSES_ICD.csv.gz", True), load("PROCEDURES_ICD.csv.gz", False)],
        ignore_index=True,
    )
    notes = pd.read_csv(
        mimic_csv_dir / "NOTEEVENTS.csv.gz",
        dtype={"HADM_ID": "string", "CATEGORY": "string"},
        compression="gzip",
        usecols=["HADM_ID", "CATEGORY"],
    )
    ds_hadm = set(notes[notes["CATEGORY"] == "Discharge summary"]["HADM_ID"].dropna())
    codes = codes[codes["HADM_ID"].isin(ds_hadm)]
    present = set(codes["ICD9_CODE"]) & set(candidate_codes)
    return len(present)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orphanet-nomenclature-pack", type=Path, required=True)
    ap.add_argument("--gem-crosswalk", type=Path, required=True)
    ap.add_argument("--mimic-csv-dir", type=Path, default=None)
    ap.add_argument(
        "--out-reference",
        type=Path,
        default=REPO_ROOT / "supplementary" / "rare_disease_reference.csv",
    )
    ap.add_argument(
        "--out-candidates",
        type=Path,
        default=REPO_ROOT / "supplementary" / "rare_icd9_candidate_codes.csv",
    )
    args = ap.parse_args()

    rows = build_reference(args.orphanet_nomenclature_pack, args.gem_crosswalk)
    write_csv(
        rows,
        args.out_reference,
        ["ORPHAcode", "icd10_code", "icd9_code", "mapping_relation"],
    )
    candidates = sorted({r["icd9_code"] for r in rows})
    write_csv(
        [{"icd9_code": c} for c in candidates], args.out_candidates, ["icd9_code"]
    )
    print(f"crosswalk rows           : {len(rows)}")
    print(f"distinct candidate ICD-9 : {len(candidates)}  (expected 992)")

    if args.mimic_csv_dir is not None:
        effective = count_effective_codes(candidates, args.mimic_csv_dir)
        print(f"rare label space (in MIMIC-III): {effective}  (expected 568)")


if __name__ == "__main__":
    main()
