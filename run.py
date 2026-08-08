# Imports
import argparse
import json
import os

import pandas
import torch

from MMACNet.utils.configuration import Config
from MMACNet.utils.import_related_ops import pandas_related_ops
from MMACNet.utils.mapper import ConfigMapper
from MMACNet.utils.misc import seed
from multiprocessing import freeze_support


def _format_param_count(value):
    return f"{value:,}" if isinstance(value, int) else str(value)


def _iter_leaf_modules(model):
    for name, module in model.named_modules():
        if name == "":
            continue
        if any(module.children()):
            continue
        yield name, module


def _print_model_summary(model, _config):
    lines = []
    total_params = 0
    total_trainable = 0

    for name, module in _iter_leaf_modules(model):
        params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable = sum(
            p.numel() for p in module.parameters(recurse=False) if p.requires_grad
        )
        if params == 0:
            continue
        total_params += params
        total_trainable += trainable
        lines.append(
            (
                name,
                module.__class__.__name__,
                _format_param_count(params),
                _format_param_count(trainable),
            )
        )

    if not lines:
        print(model)
        return

    headers = ("Layer", "Type", "Params", "Trainable")
    rows = [headers] + lines + [
        (
            "TOTAL",
            "-",
            _format_param_count(total_params),
            _format_param_count(total_trainable),
        )
    ]

    col_widths = [
        max(len(str(row[idx])) for row in rows) for idx in range(len(headers))
    ]

    def _print_row(row):
        print(
            " | ".join(
                str(col).ljust(col_widths[idx]) for idx, col in enumerate(row)
            )
        )

    separator = "-+-".join("-" * width for width in col_widths)
    _print_row(headers)
    print(separator)
    for row in lines:
        _print_row(row)
    print(separator)
    _print_row(rows[-1])


pandas_related_ops()


def _resolve_num_classes(config):
    """Ensure model.params.num_classes matches the actual label count in labels.json.

    This prevents the CUDA assertion `t >= 0 && t < n_classes` that occurs when
    a target label index exceeds the model's output dimension.
    """
    dataset_dir = config.model.params.dataset_dir
    try:
        label_file = config.dataset.params.train.label_file
    except AttributeError:
        label_file = "labels.json"
    label_path = os.path.join(dataset_dir, label_file)
    if not os.path.exists(label_path):
        print(f"[WARNING] labels.json not found at {label_path}; "
              f"using config num_classes={config.model.params.num_classes}")
        return

    with open(label_path, "r") as f:
        labels = json.load(f)
    actual_num_classes = len(labels)
    configured_num_classes = config.model.params.num_classes

    if actual_num_classes != configured_num_classes:
        print(
            f"[WARNING] num_classes mismatch: config says {configured_num_classes} "
            f"but labels.json has {actual_num_classes} labels. "
            f"Overriding num_classes to {actual_num_classes}."
        )
        config.model.params.set_value("num_classes", actual_num_classes)
    else:
        print(f"[INFO] num_classes={actual_num_classes} matches labels.json.")


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Train or test the model")
    parser.add_argument(
        "--config_path", type=str, action="store", help="Path to the config file"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether to use validation data or test data",
        default=False,
    )
    parser.add_argument(
        "--model_summary",
        action="store_true",
        help="Whether to print model summary. Note that this is supported only for "
        "models which take in a 2D input. This will be extended later",
        default=True,
    )
    args = parser.parse_args()

    # Config
    config = Config(path=args.config_path)

    if not args.test:  # Training
        # Seed
        seed(config.trainer.params.seed)

        # Auto-resolve num_classes from labels.json to prevent mismatches
        _resolve_num_classes(config)

        # Load dataset
        train_data = ConfigMapper.get_object("datasets", config.dataset.name)(
            config.dataset.params.train
        )
        val_data = ConfigMapper.get_object("datasets", config.dataset.name)(
            config.dataset.params.val
        )

        # Model
        model = ConfigMapper.get_object("models", config.model.name)(
            config.model.params
        )

        if args.model_summary:
            _print_model_summary(model, config)
        # Trainer
        trainer = ConfigMapper.get_object("trainers", config.trainer.name)(
            config.trainer.params
        )

        # Train!
        trainer.train(model, train_data, val_data)
    else:  # Test
        # Auto-resolve num_classes from labels.json to prevent mismatches
        _resolve_num_classes(config)

        # Load dataset
        test_data = ConfigMapper.get_object("datasets", config.dataset.name)(
            config.dataset.params.test
        )

        # Model
        model = ConfigMapper.get_object("models", config.model.name)(
            config.model.params
        )

        # Trainer
        trainer = ConfigMapper.get_object("trainers", config.trainer.name)(
            config.trainer.params
        )

        # Test!
        trainer.test(model, test_data)

if __name__=="__main__":
    freeze_support()
    main()
