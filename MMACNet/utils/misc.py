"""Miscellaneous utility functions."""

import copy
import itertools
import random

import numpy as np
import torch


def seed(value=42):
    """Set random seed for everything.

    Args:
        value (int): Seed
    """
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(value)


def map_dict_to_obj(dic):
    result_dic = {}
    if dic is not None:
        for k, v in dic.items():
            if isinstance(v, dict):
                result_dic[k] = map_dict_to_obj(v)
            else:
                try:
                    obj = configmapper.get_object("params", v)
                    result_dic[k] = obj
                except:
                    result_dic[k] = v
    return result_dic


def get_item_in_config(config, path):

    curr = config
    if isinstance(config, dict):
        for step in path:
            curr = curr[step]
            if curr is None:
                break
    else:
        for step in path:
            curr = curr.__getattr__(step)
            if curr is None:
                break
    return curr








def generate_grid_search_configs(main_config, grid_config, root="hyperparams"):

    locations_values_pair = {}
    init = grid_config.as_dict()

    stack = [root]
    visited = [stack[-1]]

    log_label_path = None
    hparams_path = None


    while len(stack) != 0:
        root = get_item_in_config(init, stack)
        flag = 0


        if (
            not isinstance(root, dict) and "hparams" not in stack
        ):

            if isinstance(root, list):
                locations_values_pair[
                    tuple(copy.deepcopy(stack))
                ] = root
            else:
                locations_values_pair[tuple(copy.deepcopy(stack))] = [
                    root,
                ]

            _ = stack.pop()
        else:
            if isinstance(root, list) and "hparams" in stack:
                hparams_path = copy.deepcopy(stack)
                visited.append(".".join(stack))
                stack.pop()
                continue

            if "log_label" in root.keys():
                log_label_path = copy.deepcopy(
                    stack
                    + [
                        "log_label",
                    ]
                )

            if "log_label" in root.keys():
                log_label_path = copy.deepcopy(
                    stack
                    + [
                        "log_label",
                    ]
                )
            parent = root

        for key in parent.keys():
            if (
                ".".join(
                    stack
                    + [
                        key,
                    ]
                )
                not in visited
            ):
                flag = 1
                stack.append(key)
                visited.append(".".join(stack))
                break
        if flag == 0:
            stack.pop()

    paths = list(locations_values_pair.keys())
    values = itertools.product(*list(locations_values_pair.values()))

    result_configs = []
    for value in values:
        for item_index in range(len(value)):
            curr_path = paths[item_index]
            curr_item = value[item_index]

            curr_config_item = get_item_in_config(main_config, curr_path[1:-1])
            curr_config_item.set_value(curr_path[-1], curr_item)

            log_item = get_item_in_config(main_config, log_label_path[1:-1])
            log_item.set_value(log_label_path[-1], str(len(result_configs) + 1))

            hparam_item = get_item_in_config(main_config, hparams_path[1:-1])
            hparam_item.set_value(
                hparams_path[-1],
                get_item_in_config(grid_config.hyperparams, hparams_path[1:]),
            )

        result_configs.append(copy.deepcopy(main_config))
    return result_configs




def _get_color(attr):

    attr = max(-1, min(1, attr))
    if attr > 0:
        hue = 10
        sat = 100
        lig = 100 - int(80 * attr)
    else:
        hue = 220
        sat = 100

        lig = 100 - int(-80 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)


def format_special_tokens(token):
    """Convert <> to

    Example: '<Hello>' will be converted to '#Hello' to avoid confusion
    with HTML tags.

    Args:
        token (str): The token to be formatted.
    Returns:
        (str): The formatted token.
    """
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token


def html_word_importance(words, importances):
    assert len(words) <= len(importances)
    tags = ["<div>"]

    for word_index, (word, importance) in enumerate(
        zip(words, importances[: len(words)])
    ):
        word = format_special_tokens(word)
        for character in word:
            if ord(character) >= 128:
                print(word)
                break
        bg_color = _get_color(importance)
        font_color = "white" if abs(importance) > 0.5 else "black"

        unwrapped_tag = f"""<mark style="background-color:{bg_color};\
                        opacity:1.0;line-height:1.75" title="{importance:.4f}">\
                        <font color="{font_color}"> {word} </font></mark>"""
        tags.append(unwrapped_tag)
    tags.append("</div>")

    return "".join(tags)
