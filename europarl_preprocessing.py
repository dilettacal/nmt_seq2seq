import os
import random
import re
import string
import time

import pandas as pd

from project import get_full_path
from project.utils.data.preprocessing import CustomTokenizer, clean_string, clearup, perform_refinements
from project.utils.download.europarl import maybe_download_and_extract, load_data, DATA_DIR
from project.utils.utils import convert
from settings import DATA_DIR_PREPRO, DATA_DIR_RAW

BASIC_PREPRO = get_full_path(DATA_DIR_PREPRO, "europarl")
RAW_PREPRO = get_full_path(DATA_DIR_RAW, "europarl")

### File names where preprocessed sentences are stored in ###
TRAIN_FILE = "europarl_train.tsv"
VAL_FILE = "europarl_val.tsv"
TEST = FILE = "europarl_test.tsv"

DATASET = "europarl.tsv"

MAX_EXPERIMENT_LEN = 50
BUCKETS = [10, 25, MAX_EXPERIMENT_LEN]
PRAFIX = ["Europarl.", "bitext"]


def split_data(src_sents, trg_sents, test_ratio=0.3, seed=42):
    assert len(src_sents) == len(trg_sents)
    data = list(zip(src_sents, trg_sents))

    num_samples = len(data)
    print("Total samples: ", num_samples)

    test_range = int(num_samples * test_ratio)  # test dataset 0.1
    train_range = num_samples - test_range
    random.seed(seed)  # 30
    random.shuffle(data)

    data_set = data[:train_range]
    val_set = data[train_range:]

    # create test set
    num_samples = len(data_set)
    test_range = int(num_samples * 0.1)
    train_range = num_samples - test_range

    train_set = data_set[:train_range]
    test_set = data_set[train_range:]

    print(len(test_set) + len(train_set) + len(val_set))

    train_set = list(zip(*train_set))
    val_set = list(zip(*val_set))
    test_set = list(zip(*test_set))

    return train_set, val_set, test_set


def reduce_sent_len(src_sents, trg_sents, max_len=30):
    for i, (src_sent, trg_sent) in enumerate(zip(src_sents, trg_sents)):
        if i % 10000 == 0:
            print("Sentence pairs:")
            print(src_sent, trg_sent)
        src_sent = perform_refinements(src_sent)
        trg_sent = perform_refinements(trg_sent)
        if (src_sent and trg_sent) or (src_sent!="" and trg_sent != ""):
            src_split = src_sent.split(" ")
            trg_split = trg_sent.split(" ")
            if (len(src_split) <= max_len and len(trg_split) <= max_len):
                yield src_sent, trg_sent


def store_data_to_tsv(src_sents, trg_sents, language_code):
    d = {"src": src_sents, "trg": trg_sents}
    europarl_df = pd.DataFrame(d)
    europarl_df["src_len"] = list(map(lambda x: len(x), europarl_df["src"].str.split(" ")))
    europarl_df["trg_len"] = list(map(lambda x: len(x), europarl_df["trg"].str.split(" ")))

    print(europarl_df.head(10))
    path = os.path.join(BASIC_PREPRO, language_code)
    europarl_df.to_csv(os.path.join(path, DATASET), encoding="utf-8", sep="\t", index=False)
    return path


def store_to_plain_txt(src_sents, trg_sents, language_code, file_name="europarl"):
    en_file = "{}.en".format(file_name)
    trg_file = "{}.{}".format(file_name, language_code)
    with open(os.path.join(DATA_DIR_PREPRO, "europarl", language_code, en_file)) as en_out, open(
            os.path.join(DATA_DIR_PREPRO, "europarl", language_code, trg_file)) as trg_out:
        for i, (src_sent, trg_sent) in enumerate(zip(src_sents, trg_sents)):
            if i % 10000 == 0:
                print("Sentence pair:", src_sent, " > ", trg_sent)
            en_out.write("{}\n".format(src_sent))
            en_out.write("{}\n".format(trg_sent))
    print("Done!")


### tmx preprocessing
def preprocess_tokenize_europarl_generate_tsv(language_code="de", download_if_missing=True, max_len=30, save_tsv=True):
    suffixes = ["en", language_code]
    start = time.time()

    ### check if files have been preprocessed
    files = [file for file in os.listdir(os.path.join(DATA_DIR_PREPRO, "europarl", language_code)) if
             (file.startswith("bitext.tok") and file.split(".")[-1] in suffixes)]
    if not files:
        if download_if_missing:
            print("Raw train files not available. Downloading and extracting files..")
            maybe_download_and_extract(language_code="de", tmx=True)
        try:
            from TMX2Corpus import tmx2corpus
            from TMX2Corpus.tmx2corpus import FileOutput
            ### converting tmx file
            tmx_file = os.path.join(DATA_DIR, "europarl", language_code)

            data_dir = os.path.join(DATA_DIR_PREPRO, "europarl", language_code)

            tmx2corpus.convert(os.path.join(tmx_file, "{}-{}.tmx".format(language_code, "en")),
                               output=FileOutput(path=data_dir),
                               tokenizers=[CustomTokenizer("en"),
                                           CustomTokenizer("de")])
        except ImportError:
            print("Please install tmx2corpus!")


    else:
        src_data = load_data(english=True, language_code=language_code, tmx=True)
        trg_data = load_data(english=False, language_code=language_code, tmx=True)
        reduced_corpus = list(reduce_sent_len(src_data, trg_data, max_len))
        src_sents = [pair[0] for pair in reduced_corpus]
        trg_sents = [pair[1] for pair in reduced_corpus]

        assert len(src_sents) == len(trg_sents)

        print("Dataset reduced to:", len(src_sents), len(trg_sents))
        if save_tsv:
            path_to_tsv = store_data_to_tsv(src_sents, trg_sents, language_code)
            print("tsv file saved to {}".format(path_to_tsv))

        store_to_plain_txt(src_sents, trg_sents, language_code)
        ### generate splittings
        trainset, valset, testset = split_data(src_sents, trg_sents)

        print("Storing split files...")
        store_to_plain_txt(trainset[0], trainset[1], language_code, file_name="train")
        store_to_plain_txt(valset[0], valset[1], language_code, file_name="val")
        store_to_plain_txt(testset[0], testset[1], language_code, file_name="test")

    print("Total duration: {}".format(convert(time.time() - start)))


if __name__ == '__main__':
    preprocess_tokenize_europarl_generate_tsv()
