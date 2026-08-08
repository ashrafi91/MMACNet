# `datasets/`

This directory is intentionally empty in version control (see `.gitignore`). MIMIC-III is a
credentialed, restricted-access dataset — its raw files and anything derived from patient notes
must **not** be committed to this (or any public) repository. You need your own PhysioNet
credentialed access to MIMIC-III v1.4: https://physionet.org/content/mimiciii/1.4/

## Expected layout

```
datasets/
├── mimic3/
│   ├── csv/        # raw MIMIC-III *.csv.gz exports (DIAGNOSES_ICD, NOTEEVENTS, ...)
│   └── static/      # CAML's official train/val/test HADM_ID split files
└── mimic3_full/      # created by run_preprocessing.py: train/val/test.json, labels.json,
                      # tabular_meta.json, word2vec/ (word2vec embeddings + vocab)
```

By default, configs look for the raw CSVs under `datasets/mimic3/csv` and the split files under
`datasets/mimic3/static`. Both are overridable via environment variables so you don't have to
hardcode a personal/cluster path in a config file:

```
export MIMIC_CSV_DIR=/path/to/mimic3/csv
export MIMIC_STATIC_DIR=/path/to/mimic3/static
```

Run `python run_preprocessing.py --config_path configs/preprocessing/default/mimic3_full.yml`
to populate `datasets/mimic3_full/` from the raw exports.
