"""Synthetic MIMIC-III-shaped fixtures for the unit tests.

No credentialed data is used.  Six fabricated admissions exercise every branch
the manuscript cares about:

* multiple prescription rows / admission with a non-trivial mode and mean,
* multiple microbiology rows / admission,
* an admission with no structured records at all,
* an admission with no rare code (dropped from the rare subset),
* a validation admission carrying a categorical value unseen in training.
"""

import gzip
import json
import os

import numpy as np
import pandas as pd

from MMACNet.utils.configuration import Config


ADMISSIONS = {
    "100": ["401.9", "250.00", "277.00"],
    "101": ["414.01", "277.00"],
    "102": ["585.9", "759.83"],
    "103": ["276.1", "038.9"],
    "104": ["277.00", "270.1"],
    "105": ["759.83"],
}
RARE_CANDIDATES = ["277.00", "759.83", "270.1", "253.2", "271.0"]

SPLIT = {
    "train": ["100", "102", "104"],
    "val": ["101", "105"],
    "test": ["103"],
}


PRESCRIPTIONS = {
    "100": [
        ("MAIN", "aspirin", "81mg", "10", "PO"),
        ("MAIN", "aspirin", "81mg", "20", "PO"),
        ("MAIN", "heparin", "5000u", "30", "IV"),
    ],
    "102": [
        ("MAIN", "insulin", "100u", "4", "SC"),
        ("BASE", "insulin", "100u", "6", "SC"),
        ("MAIN", "insulin", "100u", "8", "IV"),
    ],
    "104": [
        ("MAIN", "vancomycin", "1g", "100", "IV"),
        ("MAIN", "vancomycin", "1g", "200", "IV"),
    ],
    "105": [
        ("MAIN", "aspirin", "81mg", "15", "NG"),
    ],

}


MICROBIOLOGY = {
    "100": [
        ("80023", "90015", "2", "S"),
        ("80293", "90015", "4", "S"),
        ("80293", "90004", "8", "R"),
    ],
    "104": [
        ("80293", "90015", "16", "R"),
    ],

}

_NOTE_BODY = (
    "fever cough sepsis pneumonia dyspnea hypotension leukocytosis "
    "fever cough sepsis pneumonia dyspnea hypotension leukocytosis "
    "fever cough sepsis pneumonia dyspnea hypotension leukocytosis "
    "fever cough sepsis pneumonia dyspnea hypotension leukocytosis"
)


def _write_gz_csv(path, columns, rows):
    df = pd.DataFrame(rows, columns=columns)
    with gzip.open(path, "wt", newline="") as fh:
        df.to_csv(fh, index=False)


def write_mimic_csvs(csv_dir):
    os.makedirs(csv_dir, exist_ok=True)

    diag_rows, proc_rows = [], []
    for hadm, codes in ADMISSIONS.items():
        for seq, code in enumerate(codes, start=1):
            row = ("1" + hadm, hadm, str(seq), code.replace(".", ""))
            (proc_rows if code[0].isdigit() and len(code.split(".")[0]) == 2 else diag_rows).append(row)
    _write_gz_csv(
        os.path.join(csv_dir, "DIAGNOSES_ICD.csv.gz"),
        ["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"],
        diag_rows,
    )
    _write_gz_csv(
        os.path.join(csv_dir, "PROCEDURES_ICD.csv.gz"),
        ["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"],
        proc_rows or [("x", "x", "1", "00")],
    )

    note_rows = []
    for hadm in ADMISSIONS:
        note_rows.append(("1" + hadm, hadm, "Discharge summary", f"admission course {_NOTE_BODY}"))
        note_rows.append(("1" + hadm, hadm, "Nursing", "should be ignored " + _NOTE_BODY))
    _write_gz_csv(
        os.path.join(csv_dir, "NOTEEVENTS.csv.gz"),
        ["SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT"],
        note_rows,
    )

    presc_rows = [(h, *r) for h, rs in PRESCRIPTIONS.items() for r in rs]
    _write_gz_csv(
        os.path.join(csv_dir, "PRESCRIPTIONS.csv.gz"),
        ["HADM_ID", "DRUG_TYPE", "DRUG", "PROD_STRENGTH", "DOSE_VAL_RX", "ROUTE"],
        presc_rows,
    )

    micro_rows = [(h, *r) for h, rs in MICROBIOLOGY.items() for r in rs]
    _write_gz_csv(
        os.path.join(csv_dir, "MICROBIOLOGYEVENTS.csv.gz"),
        ["HADM_ID", "ORG_ITEMID", "AB_ITEMID", "DILUTION_VALUE", "INTERPRETATION"],
        micro_rows,
    )


def write_split_files(static_dir):
    os.makedirs(static_dir, exist_ok=True)
    for name, key in (
        ("train_full_hadm_ids.json", "train"),
        ("dev_full_hadm_ids.json", "val"),
        ("test_full_hadm_ids.json", "test"),
    ):
        with open(os.path.join(static_dir, name), "w") as fh:
            json.dump(SPLIT[key], fh)


def write_rare_candidates(path):
    with open(path, "w") as fh:
        fh.write("icd9_code\n")
        for code in RARE_CANDIDATES:
            fh.write(code + "\n")


def build_preprocessing_config(tmp_path, rare_subset, word2vec_min_count=1):
    csv_dir = os.path.join(tmp_path, "csv")
    static_dir = os.path.join(tmp_path, "static")
    save_dir = os.path.join(tmp_path, "out_rare" if rare_subset else "out_full")
    w2v_dir = os.path.join(save_dir, "word2vec")
    rare_csv = os.path.join(tmp_path, "rare_candidate.csv")

    write_mimic_csvs(csv_dir)
    write_split_files(static_dir)
    write_rare_candidates(rare_csv)

    params = {
        "paths": {
            "mimic_dir": csv_dir,
            "static_dir": static_dir,
            "save_dir": save_dir,
            "diagnosis_code_csv_name": "DIAGNOSES_ICD.csv.gz",
            "procedure_code_csv_name": "PROCEDURES_ICD.csv.gz",
            "noteevents_csv_name": "NOTEEVENTS.csv.gz",
            "prescriptions_csv_name": "PRESCRIPTIONS.csv.gz",
            "microbiology_events_csv_name": "MICROBIOLOGYEVENTS.csv.gz",
            "train_json_name": "train.json",
            "val_json_name": "val.json",
            "test_json_name": "test.json",
            "label_json_name": "labels.json",
            "label_freq_json_name": "label_freq.json",
        },
        "dataset_metadata": {
            "column_names": {
                "subject_id": "SUBJECT_ID",
                "hadm_id": "HADM_ID",
                "category": "CATEGORY",
                "text": "TEXT",
                "icd9_code": "ICD9_CODE",
                "labels": "LABELS",
                "drug_type": "DRUG_TYPE",
                "drug": "DRUG",
                "prod_strength": "PROD_STRENGTH",
                "dose_val_rx": "DOSE_VAL_RX",
                "route": "ROUTE",
                "org_itemid": "ORG_ITEMID",
                "ad_itemid": "AB_ITEMID",
                "dilution_value": "DILUTION_VALUE",
                "interpretation": "INTERPRETATION",
            }
        },
        "dataset_splitting_method": {
            "name": "caml_official_split",
            "params": {
                "hadm_dir": static_dir,
                "train_hadm_ids_name": "train_full_hadm_ids.json",
                "val_hadm_ids_name": "dev_full_hadm_ids.json",
                "test_hadm_ids_name": "test_full_hadm_ids.json",
            },
        },
        "clinical_note_preprocessing": {
            "to_lower": {"perform": True},
            "remove_punctuation": {"perform": True},
            "remove_numeric": {"perform": True, "replace_numerics_with_letter": None},
            "remove_stopwords": {
                "perform": True,
                "params": {
                    "stopwords_file_path": None,
                    "remove_common_medical_terms": False,
                },
            },
            "stem_or_lemmatize": {
                "perform": True,
                "params": {"stemmer_name": "nltk.WordNetLemmatizer"},
            },
            "truncate": {"perform": True, "params": {"max_length": 1500}},
        },
        "retain_section_headers": True,
        "incorrect_code_loading": False,
        "count_duplicate_codes": False,
        "code_preprocessing": {
            "top_k": 0,
            "code_type": "both",
            "add_period_in_correct_pos": {"perform": True},
            "rare_subset": rare_subset,
            "rare_candidate_codes_csv": rare_csv,
        },
        "structured": {
            "categorical_fields": [
                "DRUG_TYPE", "DRUG", "PROD_STRENGTH", "ROUTE",
                "ORG_ITEMID", "AB_ITEMID", "INTERPRETATION",
            ],
            "numerical_fields": ["DOSE_VAL_RX", "DILUTION_VALUE"],
        },
        "train_embed_with_all_split": False,
        "tokenizer": {"name": "spacetokenizer", "params": None},
        "embedding": {
            "name": "word2vec",
            "params": {
                "embedding_dir": w2v_dir,
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "word2vec_params": {
                    "vector_size": 16,
                    "min_count": word2vec_min_count,
                    "epochs": 2,
                },
            },
        },
    }


    return Config(dic=params), save_dir


def make_pipeline(tmp_path, rare_subset=True, **kw):
    from MMACNet.modules.preprocessing_pipelines import MimiciiiPreprocessingPipeline

    config, save_dir = build_preprocessing_config(tmp_path, rare_subset, **kw)
    return MimiciiiPreprocessingPipeline(config), save_dir


def write_tiny_dataset_dir(dataset_dir, encoded_split, label_vocab, tabular_meta,
                           vocab_tokens):
    """Write labels.json / tabular_meta.json / word2vec/ + one split json."""
    os.makedirs(os.path.join(dataset_dir, "word2vec"), exist_ok=True)
    with open(os.path.join(dataset_dir, "labels.json"), "w") as fh:
        json.dump(label_vocab, fh)
    with open(os.path.join(dataset_dir, "tabular_meta.json"), "w") as fh:
        json.dump(tabular_meta, fh)
    with open(os.path.join(dataset_dir, "train.json"), "w") as fh:
        json.dump(encoded_split, fh)

    tokens = ["<pad>", "<unk>"] + list(vocab_tokens)
    token_to_idx = {tok: i for i, tok in enumerate(tokens)}
    with open(os.path.join(dataset_dir, "word2vec", "token_to_idx.json"), "w") as fh:
        json.dump(token_to_idx, fh)
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(tokens), 16)).astype("float32")
    emb[0] = 0.0
    np.save(os.path.join(dataset_dir, "word2vec", "embedding_matrix.npy"), emb)


def data_common(dataset_dir, data_file="train.json", max_length=1500):
    return Config(dic={
        "column_names": {"hadm_id": "HADM_ID", "clinical_note": "TEXT", "labels": "LABELS"},
        "word2vec_dir": os.path.join(dataset_dir, "word2vec"),
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "dataset_dir": dataset_dir,
        "label_file": "labels.json",
        "max_length": max_length,
        "data_file": data_file,
    })
