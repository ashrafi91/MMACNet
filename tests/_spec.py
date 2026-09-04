"""Helpers to load the canonical spec, the configs and the manifests."""

import json
import pathlib

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]


def load_spec():
    return yaml.safe_load((REPO / "EXPERIMENT_SPEC.yaml").read_text())


def load_yaml(rel_path):
    return yaml.safe_load((REPO / rel_path).read_text())


def load_json(rel_path):
    return json.loads((REPO / rel_path).read_text())


MMACNET_CONFIGS = {
    "rare": "configs/MMACNet/MMACNet_mimic3_rare.yml",
    "full": "configs/MMACNet/MMACNet_mimic3_full.yml",
    "ablation/rare_notes": "configs/MMACNet/ablation/rare_notes.yml",
    "ablation/rare_notes_tabular": "configs/MMACNet/ablation/rare_notes_tabular.yml",
    "ablation/rare_notes_categorical": "configs/MMACNet/ablation/rare_notes_categorical.yml",
    "ablation/rare_notes_tabular_categorical": "configs/MMACNet/ablation/rare_notes_tabular_categorical.yml",
}

PREPROCESSING_CONFIGS = {
    "rare": "configs/preprocessing/mimic3_rare.yml",
    "full": "configs/preprocessing/mimic3_full.yml",
}


def model_params(cfg):
    return cfg["model"]["params"]


def trainer_params(cfg):
    return cfg["trainer"]["params"]


def data_common(cfg):
    return cfg["dataset"]["data_common"]
