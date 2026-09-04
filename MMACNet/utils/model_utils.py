"""
    Model utils
"""
import codecs
import csv
import os
import re
from collections import defaultdict

import numpy as np

from MMACNet.modules.preprocessors import CodeProcessor
from MMACNet.utils.file_loaders import load_csv_as_df, load_json
from MMACNet.utils.mapper import ConfigMapper
from MMACNet.utils.text_loggers import get_logger

logger = get_logger(__name__)


def load_lookups(
    dataset_dir,
    mimic_dir,
    static_dir,
    word2vec_dir,
    label_file="labels.json",
    version="mimic3",
):
    """
    Inputs:
        args: Input arguments
        desc_embed: true if using DR-CAML
    Outputs:
        vocab lookups, ICD code lookups, description lookup
        vector lookup
    """

    embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")
    w2ind = embedding_cls.load_vocab(word2vec_dir)
    ind2w = {i: w for w, i in w2ind.items()}


    c2ind = load_json(os.path.join(dataset_dir, label_file))
    ind2c = {i: c for c, i in c2ind.items()}


    desc_dict = load_code_descriptions(
        mimic_dir=mimic_dir, static_dir=static_dir, version=version
    )




    tabular_meta_path = os.path.join(dataset_dir, "tabular_meta.json")
    tabular_meta = None
    if os.path.exists(tabular_meta_path):
        tabular_meta = load_json(tabular_meta_path)

    dicts = {
        "ind2w": ind2w,
        "w2ind": w2ind,
        "ind2c": ind2c,
        "c2ind": c2ind,
        "desc": desc_dict,
        "tabular_meta": tabular_meta,
    }
    return dicts


def load_code_descriptions(mimic_dir, static_dir, version="mimic3"):

    reformat_fn = CodeProcessor.reformat_icd_code


    desc_dict = defaultdict(str)
    if version == "mimic2":
        mapping_path = os.path.join(static_dir, "MIMIC_ICD9_mapping")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                r = csv.reader(f)
                next(r)
                for row in r:
                    desc_dict[str(row[1])] = str(row[2])
    else:
        diag_path = os.path.join(mimic_dir, "D_ICD_DIAGNOSES.csv.gz")
        proc_path = os.path.join(mimic_dir, "D_ICD_PROCEDURES.csv.gz")
        static_path = os.path.join(static_dir, "icd9_descriptions.txt")

        if os.path.exists(diag_path):
            diag_df = load_csv_as_df(diag_path, dtype={"ICD9_CODE": str})
            for _, row in diag_df.iterrows():
                desc_dict[reformat_fn(row.ICD9_CODE, True)] = row.LONG_TITLE

        if os.path.exists(proc_path):
            proc_df = load_csv_as_df(proc_path, dtype={"ICD9_CODE": str})
            for _, row in proc_df.iterrows():
                desc_dict[reformat_fn(row.ICD9_CODE, True)] = row.LONG_TITLE

        if os.path.exists(static_path):
            with open(static_path, "r") as labelfile:
                for row in labelfile:
                    row = row.rstrip().split()
                    if not row:
                        continue
                    code = row[0]
                    if code not in desc_dict:
                        desc_dict[code] = " ".join(row[1:])

    if not desc_dict:
        logger.warning(
            "No ICD-9 description sources found under %s / %s. The "
            "description regulariser (lmbda > 0) needs D_ICD_DIAGNOSES.csv.gz, "
            "D_ICD_PROCEDURES.csv.gz and icd9_descriptions.txt.",
            mimic_dir,
            static_dir,
        )
    return desc_dict


def pad_desc_vecs(desc_vecs):


    desc_len = max([len(dv) for dv in desc_vecs])
    pad_vecs = []
    for vec in desc_vecs:
        pad_vecs.append(vec + [0] * (desc_len - len(vec)))
    return pad_vecs


def _readString(f, code):

    s = str()
    c = f.read(1)
    value = ord(c)

    while value != 10 and value != 32:
        if 0x00 < value < 0xBF:
            continue_to_read = 0
        elif 0xC0 < value < 0xDF:
            continue_to_read = 1
        elif 0xE0 < value < 0xEF:
            continue_to_read = 2
        elif 0xF0 < value < 0xF4:
            continue_to_read = 3
        else:
            raise RuntimeError("not valid utf-8 code")

        i = 0



        temp = bytes()
        temp = temp + c

        while i < continue_to_read:
            temp = temp + f.read(1)
            i += 1

        temp = temp.decode(code)
        s = s + temp

        c = f.read(1)
        value = ord(c)

    return s


import struct


def _readFloat(f):
    bytes4 = f.read(4)
    f_num = struct.unpack("f", bytes4)[0]
    return f_num


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()


    if embedding_path.find(".bin") != -1:
        with open(embedding_path, "rb") as f:
            wordTotal = int(_readString(f, "utf-8"))
            embedd_dim = int(_readString(f, "utf-8"))

            for i in range(wordTotal):
                word = _readString(f, "utf-8")


                word_vector = []
                for j in range(embedd_dim):
                    word_vector.append(_readFloat(f))
                word_vector = np.array(word_vector, np.float)

                f.read(1)

                embedd_dict[word] = word_vector

    else:
        with codecs.open(embedding_path, "r", "UTF-8") as file:
            for line in file:

                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = re.split(r"\s+", line)
                if len(tokens) == 2:
                    continue
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:

                    if embedd_dim + 1 != len(tokens):
                        continue
                embedd = np.zeros([1, embedd_dim])
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd

    return embedd_dict, embedd_dim
