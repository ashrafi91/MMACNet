import os

import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from MMACNet.utils.file_loaders import load_csv_as_df, load_json, save_json
from MMACNet.utils.mapper import ConfigMapper
from MMACNet.utils.text_loggers import get_logger

logger = get_logger(__name__)


@ConfigMapper.map("datasets", "base_dataset")
class BaseDataset(Dataset):
    def __init__(self, config):
        self._config = config

        # Load vocab (dict of {word: idx})
        embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")
        self.vocab = embedding_cls.load_vocab(self._config.word2vec_dir)
        self.vocab_size = len(self.vocab)
        assert self.vocab_size == max(self.vocab.values()) + 1
        self.pad_idx = self.vocab[self._config.pad_token]
        self.unk_idx = self.vocab[self._config.unk_token]
        self.inv_vocab = {i: w for w, i in self.vocab.items()}

        # Load labels (dict of {code: idx})
        label_path = os.path.join(
            self._config.dataset_dir, self._config.label_file
        )
        self.all_labels = load_json(label_path)
        self.num_labels = len(self.all_labels)
        assert self.num_labels == max(self.all_labels.values()) + 1
        self.inv_labels = {i: c for c, i in self.all_labels.items()}
        logger.debug(
            "Loaded {} ICD code labels from {}".format(
                self.num_labels, label_path
            )
        )

        # To-do: This class currently deals with only JSON files. We can extend
        # this to deal with other file types (.csv, .xlsx, etc.).

        # Load data (JSON)
        data_path = os.path.join(
            self._config.dataset_dir, self._config.data_file
        )
        self.df = pd.DataFrame.from_dict(load_json(data_path))
        logger.info(
            "Loaded dataset from {} ({} examples)".format(
                data_path, self.df.shape
            )
        )
        self.extra_feature_cols = self._identify_extra_feature_columns()
        self.tabular_meta = None
        if self.extra_feature_cols:
            self.tabular_meta = self._load_or_create_tabular_meta()
        self.categorical_feature_order = (
            self.tabular_meta.get("categorical_order", [])
            if self.tabular_meta
            else []
        )
        self.numeric_feature_order = (
            self.tabular_meta.get("numerical_order", [])
            if self.tabular_meta
            else []
        )
        self.has_tabular_features = bool(
            self.categorical_feature_order or self.numeric_feature_order
        )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clinical_note = row[self._config.column_names.clinical_note]
        label_val = row[self._config.column_names.labels]
        if isinstance(label_val, str):
            codes = [label_val]
        elif isinstance(label_val, (list, tuple)) and label_val:
            codes = [str(label_val[0])]
        else:
            codes = []

        # Note (list) -> word idxs (UNK is assigned at the last word)
        token_idxs = self.encode_tokens(clinical_note)

        # ICD codes -> binary labels
        labels = self.encode_labels(codes)
        one_hot_labels = np.zeros(self.num_labels, dtype=np.int32)
        for l in labels:
            one_hot_labels[l] = 1
        tabular_features = (
            self._prepare_tabular_features(row) if self.has_tabular_features else None
        )

        if self.has_tabular_features:
            return (token_idxs, tabular_features, one_hot_labels)
        return (token_idxs, one_hot_labels)

    def encode_tokens(self, tokens):
        """Convert list of words into list of token idxs, and truncate"""
        token_idxs = [
            self.vocab[w] if w in self.vocab else self.unk_idx for w in tokens
        ]
        token_idxs = token_idxs[: self._config.max_length]
        return token_idxs

    def decode_tokens(self, token_idxs):
        """Convert list of token idxs into list of words"""
        return [self.inv_vocab[idx] for idx in token_idxs]

    def encode_labels(self, codes):
        """Convert list of ICD codes into labels"""
        return [self.all_labels[c] for c in codes]

    def decode_labels(self, labels):
        """Convert labels into list of ICD codes"""
        return [self.inv_labels[l] for l in labels]

    def collate_fn(self, examples):
        """Concatenate examples into note, optional tabular, and label tensors"""
        if not examples:
            raise ValueError("Cannot collate empty batch.")
        first_example = examples[0]
        if not isinstance(first_example, tuple):
            raise ValueError("Expected dataset examples to be tuples.")

        has_tabular = len(first_example) == 3

        if has_tabular:
            notes, tabular_features, labels = zip(*examples)
        elif len(first_example) == 2:
            notes, labels = zip(*examples)
            tabular_features = None
        else:
            raise ValueError(
                "Dataset examples must be (text, label) or (text, tabular, label)."
            )

        max_note_len = max(map(len, notes))
        padded_notes = [
            note + [self.pad_idx] * (max_note_len - len(note)) for note in notes
        ]
        notes_tensor = torch.tensor(padded_notes)
        labels_tensor = torch.tensor(labels)

        if not has_tabular or tabular_features is None:
            return notes_tensor, labels_tensor

        categorical_feats = (
            torch.tensor(
                [feat["categorical"] for feat in tabular_features],
                dtype=torch.long,
            )
            if self.categorical_feature_order
            else None
        )
        numeric_feats = (
            torch.tensor(
                [feat["numerical"] for feat in tabular_features],
                dtype=torch.float,
            )
            if self.numeric_feature_order
            else None
        )

        inputs = {"text": notes_tensor}
        if categorical_feats is not None:
            inputs["categorical"] = categorical_feats
        if numeric_feats is not None:
            inputs["numerical"] = numeric_feats

        return inputs, labels_tensor

    def _identify_extra_feature_columns(self):
        base_cols = {
            self._config.column_names.hadm_id,
            self._config.column_names.clinical_note,
            self._config.column_names.labels,
        }
        return [
            col for col in self.df.columns if col not in base_cols
        ]

    def _load_or_create_tabular_meta(self):
        meta_path = os.path.join(
            self._config.dataset_dir, "tabular_meta.json"
        )
        if os.path.exists(meta_path):
            logger.info(f"Loading tabular metadata from {meta_path}")
            return load_json(meta_path)
        meta = self._create_tabular_meta()
        save_json(meta, meta_path)
        logger.info(f"Saved tabular metadata to {meta_path}")
        return meta

    def _create_tabular_meta(self):
        categorical_meta = {}
        numerical_meta = {}
        categorical_order = []
        numerical_order = []

        for col in self.extra_feature_cols:
            series = self.df[col]
            if self._is_numeric_series(series):
                clean_series = pd.to_numeric(series, errors="coerce")
                mean = float(clean_series.mean(skipna=True))
                std = float(clean_series.std(skipna=True))
                if math.isnan(mean):
                    mean = 0.0
                if math.isnan(std) or std == 0.0:
                    std = 1.0
                numerical_meta[col] = {"mean": mean, "std": std}
                numerical_order.append(col)
            else:
                values = (
                    series.dropna()
                    .astype(str)
                    .apply(lambda v: v.strip() if isinstance(v, str) else v)
                    .tolist()
                )
                unique_values = sorted(set(values))
                if not unique_values:
                    continue
                mapping = {
                    val: idx + 1 for idx, val in enumerate(unique_values)
                }
                categorical_meta[col] = {
                    "mapping": mapping,
                    "unk_index": 0,
                    "num_classes": len(unique_values) + 1,
                }
                categorical_order.append(col)

        return {
            "categorical": categorical_meta,
            "numerical": numerical_meta,
            "categorical_order": categorical_order,
            "numerical_order": numerical_order,
        }

    def _prepare_tabular_features(self, row):
        categorical = []
        for col in self.categorical_feature_order:
            meta = self.tabular_meta["categorical"][col]
            mapping = meta["mapping"]
            value = row[col]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                categorical.append(meta["unk_index"])
                continue
            str_val = str(value).strip()
            categorical.append(mapping.get(str_val, meta["unk_index"]))

        numerical = []
        for col in self.numeric_feature_order:
            meta = self.tabular_meta["numerical"][col]
            value = row[col]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                value = meta["mean"]
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = meta["mean"]
            normalized = (value - meta["mean"]) / meta["std"]
            numerical.append(normalized)

        return {"categorical": categorical, "numerical": numerical}

    def _is_numeric_series(self, series):
        numeric_series = pd.to_numeric(series, errors="coerce")
        non_null_ratio = numeric_series.notnull().mean()
        return non_null_ratio > 0.8
