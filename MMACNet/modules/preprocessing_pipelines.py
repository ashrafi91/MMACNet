"""MIMIC-III preprocessing pipeline for MMAC-Net.

This is the "final preprocessing code" referenced by the manuscript
(Sections 3.5, 3.6 and 5.1).  It differs from a plain CAML-style pipeline in
exactly the ways the manuscript describes:

* **One sample per admission.**  All ICD-9 codes assigned to a hospital
  admission become a single multi-hot label vector.  Codes are never exploded
  into one-row-per-code.

* **All structured records are aggregated.**  An admission carries many
  ``PRESCRIPTIONS`` and ``MICROBIOLOGYEVENTS`` rows.  Every row is folded into
  one fixed-width feature vector: categorical fields collapse to their
  per-admission mode over *all* rows, numerical fields to their per-admission
  mean over *all* rows.

* **Train-only fitting.**  The token vocabulary / Word2Vec model, the
  categorical vocabularies and the numerical z-score scalers are all fit on the
  training split alone; categories unseen in training map to a reserved index.

* **Rare subset.**  When ``code_preprocessing.rare_subset`` is true an
  admission is kept iff at least one of its codes is in the Orphanet-derived
  candidate set, and its label vector is restricted to those rare codes
  (Section 3.1).

Every numeric choice here is mirrored in ``EXPERIMENT_SPEC.yaml`` and checked by
``tests/test_experiment_settings_consistency.py``.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from MMACNet.modules.preprocessors import ClinicalNotePreprocessor, CodeProcessor
from MMACNet.utils.file_loaders import load_csv_as_df, save_json
from MMACNet.utils.mapper import ConfigMapper
from MMACNet.utils.text_loggers import get_logger

logger = get_logger(__name__)
tqdm.pandas()


DEFAULT_CATEGORICAL_FIELDS = (
    "DRUG_TYPE",
    "DRUG",
    "PROD_STRENGTH",
    "ROUTE",
    "ORG_ITEMID",
    "AB_ITEMID",
    "INTERPRETATION",
)
DEFAULT_NUMERICAL_FIELDS = ("DOSE_VAL_RX", "DILUTION_VALUE")


def _series_mode(series):
    """Most frequent non-null value across *all* rows (deterministic tie-break).

    Returns ``None`` when the admission has no value for this field.  Using the
    mode -- not ``first()`` -- is what makes the aggregation depend on every
    record rather than on record order.
    """
    clean = series.dropna()
    if clean.empty:
        return None
    modes = clean.astype(str).str.strip()
    modes = modes[modes != ""]
    if modes.empty:
        return None
    counts = modes.value_counts()
    top = counts[counts == counts.max()].index
    return sorted(top)[0]


def _series_mean(series):
    """Mean of the numeric values across *all* rows; ``None`` if none parse."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


@ConfigMapper.map("preprocessing_pipelines", "mimic_iii_preprocessing_pipeline")
@ConfigMapper.map(
    "preprocessing_pipelines", "mimic_iii_preprocessing_pipeline_all_codes"
)
class MimiciiiPreprocessingPipeline:
    def __init__(self, config):
        self.config = config
        self.MIMIC_DIR = config.paths.mimic_dir
        self.SAVE_DIR = config.paths.save_dir
        self.cols = config.dataset_metadata.column_names
        self.clinical_note_config = config.clinical_note_preprocessing
        self.code_config = config.code_preprocessing

        os.makedirs(self.MIMIC_DIR, exist_ok=True)
        os.makedirs(self.SAVE_DIR, exist_ok=True)

        self.clinical_note_preprocessor = ClinicalNotePreprocessor(
            self.clinical_note_config
        )
        self.code_preprocessor = CodeProcessor(self.code_config)

        self.split_data = ConfigMapper.get_object(
            "dataset_splitters", config.dataset_splitting_method.name
        )(config.dataset_splitting_method.params)
        self.tokenizer = ConfigMapper.get_object(
            "tokenizers", config.tokenizer.name
        )(config.tokenizer.params)
        self.embedder = ConfigMapper.get_object(
            "embeddings", config.embedding.name
        )(config.embedding.params)


        structured = getattr(config, "structured", None)
        self.categorical_fields = list(
            getattr(structured, "categorical_fields", DEFAULT_CATEGORICAL_FIELDS)
            if structured
            else DEFAULT_CATEGORICAL_FIELDS
        )
        self.numerical_fields = list(
            getattr(structured, "numerical_fields", DEFAULT_NUMERICAL_FIELDS)
            if structured
            else DEFAULT_NUMERICAL_FIELDS
        )

        self.rare_subset = bool(
            getattr(self.code_config, "rare_subset", False)
        )
        self.rare_candidate_csv = getattr(
            self.code_config, "rare_candidate_codes_csv", None
        )
        self.top_k = int(getattr(self.code_config, "top_k", 0))
        self.count_duplicate_codes = bool(
            getattr(config, "count_duplicate_codes", False)
        )

        self._csv_str = {"dtype": "string"}


    def _mimic_path(self, name):
        return os.path.join(self.MIMIC_DIR, name)

    def extract_codes(self):
        """Return a [HADM_ID, ICD9_CODE] frame with reformatted codes."""
        hcol, ccol = self.cols.hadm_id, self.cols.icd9_code
        diag = load_csv_as_df(
            self._mimic_path(self.config.paths.diagnosis_code_csv_name),
            dtype={hcol: "string", ccol: "string"},
        )
        proc = load_csv_as_df(
            self._mimic_path(self.config.paths.procedure_code_csv_name),
            dtype={hcol: "string", ccol: "string"},
        )
        diag[ccol] = diag[ccol].apply(
            lambda x: str(self.code_preprocessor(str(x), True))
        )
        proc[ccol] = proc[ccol].apply(
            lambda x: str(self.code_preprocessor(str(x), False))
        )
        code_type = self.code_config.code_type
        assert code_type in ("diagnosis", "procedure", "both")
        if code_type == "diagnosis":
            code_df = diag
        elif code_type == "procedure":
            code_df = proc
        else:
            code_df = pd.concat([diag, proc], ignore_index=True)
        return code_df[[hcol, ccol]].dropna(subset=[hcol, ccol])

    def load_notes(self):
        """Discharge summaries, concatenated + cleaned, one row per admission."""
        hcol, tcol, cat = self.cols.hadm_id, self.cols.text, self.cols.category
        df = load_csv_as_df(
            self._mimic_path(self.config.paths.noteevents_csv_name),
            dtype={hcol: "string", tcol: "string"},
        )
        df = df[df[cat] == "Discharge summary"][[hcol, tcol]].dropna()
        merged = (
            df.groupby(hcol)[tcol]
            .apply(lambda texts: " ".join(texts))
            .reset_index()
        )
        logger.info("Cleaning %d discharge summaries", len(merged))
        merged[tcol] = [
            self.clinical_note_preprocessor(t)
            for t in tqdm(merged[tcol].tolist(), desc="clean notes")
        ]
        return merged

    def load_prescriptions(self):
        hcol = self.cols.hadm_id
        keep = [
            hcol,
            self.cols.drug_type,
            self.cols.drug,
            self.cols.prod_strength,
            self.cols.dose_val_rx,
            self.cols.route,
        ]
        df = load_csv_as_df(
            self._mimic_path(self.config.paths.prescriptions_csv_name),
            dtype={c: "string" for c in keep},
        )
        return df[[c for c in keep if c in df.columns]]

    def load_microbiology(self):
        hcol = self.cols.hadm_id
        keep = [
            hcol,
            self.cols.org_itemid,
            self.cols.ad_itemid,
            self.cols.dilution_value,
            self.cols.interpretation,
        ]
        df = load_csv_as_df(
            self._mimic_path(self.config.paths.microbiology_events_csv_name),
            dtype={c: "string" for c in keep},
        )
        return df[[c for c in keep if c in df.columns]]


    def aggregate_structured(self, prescriptions_df, microbiology_df):
        """Fold *all* prescription + microbiology rows for an admission into a
        single fixed-width row (Section 3.5).

        Categorical fields -> per-admission mode over every row.
        Numerical fields   -> per-admission mean over every row.
        """
        hcol = self.cols.hadm_id
        frames = []
        for source in (prescriptions_df, microbiology_df):
            if source is None or source.empty:
                continue
            present_cat = [c for c in self.categorical_fields if c in source.columns]
            present_num = [c for c in self.numerical_fields if c in source.columns]
            agg_spec = {}
            for c in present_cat:
                agg_spec[c] = _series_mode
            for c in present_num:
                agg_spec[c] = _series_mean
            if not agg_spec:
                continue
            grouped = source.groupby(hcol).agg(agg_spec).reset_index()
            frames.append(grouped)

        if not frames:
            return pd.DataFrame(columns=[hcol])

        out = frames[0]
        for frame in frames[1:]:
            out = out.merge(frame, on=hcol, how="outer")

        for c in self.categorical_fields + self.numerical_fields:
            if c not in out.columns:
                out[c] = None
        out = out[[hcol] + self.categorical_fields + self.numerical_fields]
        return self._normalise_structured_dtypes(out)

    def _normalise_structured_dtypes(self, frame):
        """Categorical -> plain ``object`` with ``None`` for missing;
        numerical -> ``float64`` with ``NaN`` for missing.  Keeps pandas
        extension dtypes (``string`` / ``pd.NA``) from leaking downstream."""
        for c in self.categorical_fields:
            if c in frame.columns:
                col = frame[c].astype("object")
                frame[c] = col.where(col.notna(), None)
        for c in self.numerical_fields:
            if c in frame.columns:
                frame[c] = pd.to_numeric(frame[c], errors="coerce").astype(float)
        return frame

    def build_admission_table(self, code_df, notes_df, structured_df):
        """One row per admission: TEXT, LABELS (list), structured columns."""
        hcol, ccol = self.cols.hadm_id, self.cols.icd9_code
        tcol, lcol = self.cols.text, self.cols.labels

        if self.count_duplicate_codes:
            codes = code_df.groupby(hcol)[ccol].apply(list)
        else:
            codes = code_df.groupby(hcol)[ccol].apply(
                lambda s: list(dict.fromkeys(s))
            )
        codes = codes.reset_index().rename(columns={ccol: lcol})


        table = notes_df.merge(codes, on=hcol, how="inner")
        table = table.merge(structured_df, on=hcol, how="left")
        for c in self.categorical_fields + self.numerical_fields:
            if c not in table.columns:
                table[c] = None
        table = self._normalise_structured_dtypes(table)
        table = table.sort_values(hcol).reset_index(drop=True)
        return table[[hcol, tcol, lcol] + self.categorical_fields + self.numerical_fields]


    def load_rare_candidates(self):
        path = self.rare_candidate_csv
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                "code_preprocessing.rare_candidate_codes_csv must point at the "
                "Orphanet-derived candidate list "
                "(supplementary/rare_icd9_candidate_codes.csv). Regenerate it "
                "with scripts/build_rare_subset.py."
            )
        df = pd.read_csv(path, dtype=str)
        col = "icd9_code" if "icd9_code" in df.columns else df.columns[0]
        return set(df[col].dropna().astype(str).str.strip())

    def apply_rare_filter(self, table, rare_codes):
        """Keep admissions with >=1 rare code; restrict labels to the rare set."""
        lcol = self.cols.labels
        rare_codes = set(rare_codes)
        mask = table[lcol].apply(lambda codes: bool(set(codes) & rare_codes))
        table = table[mask].copy()
        table[lcol] = table[lcol].apply(
            lambda codes: [c for c in codes if c in rare_codes]
        )
        return table.reset_index(drop=True)


    def fit_label_vocab(self, *split_tables):
        """Corpus-level ICD-9 label vocabulary ``{code: idx}``.

        Order is deterministic: descending corpus frequency, then code string.
        ``top_k`` (0 = keep all) mirrors CAML's ``TopKCodes``.
        """
        lcol = self.cols.labels
        counts = Counter()
        for table in split_tables:
            for codes in table[lcol]:
                counts.update(codes)
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if self.top_k and self.top_k > 0:
            ordered = ordered[: self.top_k]
        return {code: idx for idx, (code, _) in enumerate(ordered)}, dict(counts)

    def fit_tabular_meta(self, train_table):
        """Categorical vocabularies + numerical z-score scalers, TRAIN ONLY."""
        categorical, numerical = {}, {}
        for col in self.categorical_fields:
            values = (
                train_table[col]
                .dropna()
                .astype(str)
                .map(str.strip)
            )
            values = sorted(v for v in set(values) if v != "")
            mapping = {v: i + 1 for i, v in enumerate(values)}
            categorical[col] = {
                "mapping": mapping,
                "unk_index": 0,
                "num_classes": len(mapping) + 1,
            }
        for col in self.numerical_fields:
            series = pd.to_numeric(train_table[col], errors="coerce").dropna()
            mean = float(series.mean()) if not series.empty else 0.0
            std = float(series.std()) if not series.empty else 1.0
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            if not np.isfinite(mean):
                mean = 0.0
            numerical[col] = {"mean": mean, "std": std}
        return {
            "categorical": categorical,
            "numerical": numerical,
            "categorical_order": list(self.categorical_fields),
            "numerical_order": list(self.numerical_fields),
        }


    def encode_split(self, table, tokenize=True):
        """Turn a split table into a ``{column: [values]}`` dict for JSON."""
        hcol, tcol, lcol = self.cols.hadm_id, self.cols.text, self.cols.labels
        out = {
            hcol: [str(h) for h in table[hcol].tolist()],
            lcol: [list(map(str, codes)) for codes in table[lcol].tolist()],
        }
        texts = table[tcol].fillna("").tolist()
        out[tcol] = (
            self.tokenizer.tokenize_list(texts) if tokenize else texts
        )
        for col in self.categorical_fields:
            out[col] = [
                "" if pd.isna(v) else str(v).strip() for v in table[col].tolist()
            ]
        for col in self.numerical_fields:
            out[col] = [
                None if pd.isna(v) else float(v) for v in table[col].tolist()
            ]
        return out


    def preprocess(self):
        hcol = self.cols.hadm_id

        logger.info("Loading MIMIC-III sources")
        code_df = self.extract_codes()
        notes_df = self.load_notes()
        prescriptions_df = self.load_prescriptions()
        microbiology_df = self.load_microbiology()

        logger.info("Aggregating all prescription + microbiology records")
        structured_df = self.aggregate_structured(prescriptions_df, microbiology_df)

        logger.info("Building one-row-per-admission table")
        table = self.build_admission_table(code_df, notes_df, structured_df)

        if self.rare_subset:
            rare_codes = self.load_rare_candidates()
            table = self.apply_rare_filter(table, rare_codes)
            logger.info("Rare subset: %d admissions retained", len(table))

        logger.info("Applying the benchmark train/val/test split")
        train_t, val_t, test_t = self.split_data(table, hcol)

        label_vocab, label_counts = self.fit_label_vocab(train_t, val_t, test_t)
        save_json(
            label_vocab,
            os.path.join(self.SAVE_DIR, self.config.paths.label_json_name),
        )
        logger.info("Label space: %d ICD-9 codes", len(label_vocab))
        freq_name = getattr(self.config.paths, "label_freq_json_name", None)
        if freq_name:
            save_json(label_counts, os.path.join(self.SAVE_DIR, freq_name))

        tabular_meta = self.fit_tabular_meta(train_t)
        save_json(
            tabular_meta, os.path.join(self.SAVE_DIR, "tabular_meta.json")
        )

        splits = {
            self.config.paths.train_json_name: train_t,
            self.config.paths.val_json_name: val_t,
            self.config.paths.test_json_name: test_t,
        }
        for fname, split_table in splits.items():
            encoded = self.encode_split(split_table, tokenize=True)
            save_json(encoded, os.path.join(self.SAVE_DIR, fname))

        logger.info("Training Word2Vec on the training split")
        if getattr(self.config, "train_embed_with_all_split", False):
            corpus_tables = [train_t, val_t, test_t]
        else:
            corpus_tables = [train_t]
        corpus = []
        for split_table in corpus_tables:
            corpus.extend(
                self.tokenizer.tokenize_list(
                    split_table[self.cols.text].fillna("").tolist()
                )
            )
        self.embedder.train(corpus)
        logger.info("Preprocessing complete -> %s", self.SAVE_DIR)
