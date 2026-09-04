import os

import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from MMACNet.utils.file_loaders import load_json
from MMACNet.utils.mapper import ConfigMapper
from MMACNet.utils.text_loggers import get_logger

logger = get_logger(__name__)


@ConfigMapper.map("datasets", "base_dataset")
class BaseDataset(Dataset):
    def __init__(self, config):
        self._config = config


        embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")
        self.vocab = embedding_cls.load_vocab(self._config.word2vec_dir)
        self.vocab_size = len(self.vocab)
        assert self.vocab_size == max(self.vocab.values()) + 1
        self.pad_idx = self.vocab[self._config.pad_token]
        self.unk_idx = self.vocab[self._config.unk_token]
        self.inv_vocab = {i: w for w, i in self.vocab.items()}


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
        elif isinstance(label_val, (list, tuple, np.ndarray)):
            codes = [str(c) for c in label_val]
        else:
            codes = []


        token_idxs = self.encode_tokens(clinical_note)


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
        """Convert list of ICD codes into label indices.

        Codes outside the fitted label space (e.g. dropped by ``top_k``) are
        skipped rather than raising, so a restricted label vocabulary stays
        usable.
        """
        return [self.all_labels[c] for c in codes if c in self.all_labels]

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
        raise FileNotFoundError(
            f"tabular_meta.json not found at {meta_path}. It is fit on the "
            "training split by run_preprocessing.py and must not be derived "
            "from val/test data. Run the preprocessing pipeline first."
        )

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
