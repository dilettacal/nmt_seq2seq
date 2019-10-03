"""
This file contains functions to persist and split data.
"""

import os
import random
from settings import SEED


def split_data(src_sents, trg_sents, val_ratio=0.1, train_ratio=0.8, seed=SEED):
    """
    Split the source and target sentences using the provided ratios and seed
    Default: 80, 10, 10
    :param src_sents: list containing only the source sentences
    :param trg_sents: list containing only the target sentences
    :param val_ratio: validation ratio
    :param train_ratio: training ration
    :param seed: splits on the provided feed, default see settings.py
    :return: splits
    """

    assert len(src_sents) == len(trg_sents)
    data = list(zip(src_sents, trg_sents))

    num_samples = len(data)
    print("Total samples: ", num_samples)

    print("Shuffling data....")
    random.seed(seed)  # 30
    random.shuffle(data)

    if isinstance(val_ratio, int):
        print("Fixed validation/test ratio:", val_ratio)
        val_set = data[:val_ratio]
        test_set = data[val_ratio:val_ratio+val_ratio]
        train_set = data[val_ratio+val_ratio:]
    else:
        train_end = int(train_ratio * num_samples)
        validate_end = int(val_ratio * num_samples) + train_end
        train_set = data[:train_end]
        val_set = data[train_end:validate_end]
        test_set = data[validate_end:]

    print("Total train:", len(train_set))
    print("Total validation:", len(val_set))
    print("Total test:", len(test_set))
    print("All togheter:", len(test_set) + len(train_set) + len(val_set))

    samples = train_set[:5] + val_set[:5] + test_set[:5]

    train_set = list(zip(*train_set))
    val_set = list(zip(*val_set))
    test_set = list(zip(*test_set))

    samples_set = list(zip(*samples))
    return train_set, val_set, test_set, samples_set


def persist_txt(lines, store_path, file_name, exts):
    """
    Stores the given lines
    :param lines: bilingual list of sentences
    :param store_path: path to store the file in
    :param file_name:
    :param exts: tuple containing the extensions, should match the line order, default: (.en, lang_code)
    :return:
    """
    with open(os.path.join(store_path, file_name + exts[0]), mode="w", encoding="utf-8") as src_out_file, \
            open(os.path.join(store_path, file_name + exts[1]), mode="w", encoding="utf-8") as trg_out_file:
        if len(lines) == 2:
            lines = list(zip(lines[0], lines[1]))
            for src, trg in lines:
                src_out_file.write("{}\n".format(src))
                trg_out_file.write("{}\n".format(trg))