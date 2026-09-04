# Supplementary materials

Files referenced by the manuscript (Section 3.1, Section 7.4, Appendix B).

| File | Status | Description |
|------|--------|-------------|
| `rare_disease_reference.csv` | **regenerated** (`scripts/build_rare_subset.py`) | Complete Orphanet-derived crosswalk: every rare ICD-9 code with its ORPHAcode, ICD-10 code and mapping relation. Appendix B. Needs the licensed Orphanet nomenclature pack + CMS GEMs. |
| `rare_disease_reference.SCHEMA.csv` | committed | Header-only template showing the four columns of `rare_disease_reference.csv`. |
| `rare_icd9_candidate_codes.csv` | committed | The 992 candidate rare ICD-9 codes (the ICD-9 column of the crosswalk). Ships in the repo because it contains no patient data. |
| `rare_code_top_ngrams.csv` | **regenerated** (`scripts/export_rare_code_top_ngrams.py`) | Highest-weighted n-grams per rare code, aggregated over the test split, from the trained checkpoint. Section 7.4 / Appendix B. |
| `rare_code_top_ngrams.SCHEMA.csv` | committed | Header-only template for `rare_code_top_ngrams.csv`. |

## Label counts

| Quantity | Value | Where it is fixed |
|----------|-------|-------------------|
| Candidate rare ICD-9 codes (Orphanet -> GEM crosswalk) | **992** | `rare_icd9_candidate_codes.csv`, `EXPERIMENT_SPEC.yaml: dataset.rare_subset_construction.candidate_icd9_codes` |
| Rare label space `num_classes` (candidates that occur in MIMIC-III discharge-summary admissions) | **568** | manuscript Table 2 / Table 4, `EXPERIMENT_SPEC.yaml: dataset.rare.distinct_icd9_codes`, `configs/MMACNet/MMACNet_mimic3_rare.yml`, `datasets/mimic3_rare/labels.json` (after preprocessing) |
| Full label space `num_classes` | **8930** | manuscript Table 2 / Table 4, `EXPERIMENT_SPEC.yaml: dataset.all.distinct_icd9_codes`, `configs/MMACNet/MMACNet_mimic3_full.yml`, `datasets/mimic3_full/labels.json` |

`tests/test_label_counts_consistency.py` checks that 568 and 8930 appear
identically across the spec, the configs and the label manifests.

## Regenerating the rare subset

```
python scripts/build_rare_subset.py \
    --orphanet-nomenclature-pack /path/to/orphanet_dec2025 \
    --gem-crosswalk /path/to/cms_icd10_to_icd9_gem.txt \
    --mimic-csv-dir datasets/mimic3/csv \
    --out-reference supplementary/rare_disease_reference.csv \
    --out-candidates supplementary/rare_icd9_candidate_codes.csv
```

The script is deterministic: given the December 2025 Orphanet pack and the CMS
GEMs it reproduces the 992-code candidate list and, after intersecting with the
MIMIC-III admissions that carry a discharge summary, the 568-code rare label
space.
