import os
import time
import numpy as np

import pandas as pd


from project import get_full_path
from project.utils.data.preprocessing import preprocess_corpus, generate_splits_from_datasets
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


def preprocess_tokenize_europarl_generate_tsv(language_code="de", download_if_missing=True, tmx=True):

    suffixes = ["en", language_code]
    files = []
    if tmx:
        files = [file for file in os.listdir(os.path.join(DATA_DIR, language_code)) if (file.startswith("bitext") and file.split(".")[-1] in suffixes) and "tok" not in file]
        print(files)

    if files:
        src_data = load_data(english=True, language_code=language_code, tmx=tmx)
        trg_data = load_data(english=False, language_code=language_code, tmx=tmx)
    else:
        if download_if_missing:
            print("Raw train files not available. Downloading and extracting files..")
            raw_file = maybe_download_and_extract(language_code="de", tmx=tmx)
            print("File extracted:", raw_file)
            try:
                import tmx2corpus
                from tmx2corpus import FileOutput
                ### converting tmx file
                data_dir = os.path.join(DATA_DIR, language_code)
                tmx2corpus.convert(os.path.join(data_dir, raw_file), output=FileOutput(path=data_dir))
            except ImportError: print("Please install tmx2corpus!")
            src_data = load_data(english=True, language_code=language_code, tmx=tmx)
            trg_data = load_data(english=False, language_code=language_code, tmx=tmx)
        else:
            raise
    print("Source data overview:", src_data[:2])
    print("Target data overview:", trg_data[:2])
    if not os.path.isfile(os.path.join(BASIC_PREPRO, language_code, DATASET)):
        start = time.time()
        preprocessed_sents = list(preprocess_corpus(src_data, trg_data, language_code))
        src_sents = [pair[0] for pair in preprocessed_sents]
        trg_sents = [pair[1] for pair in preprocessed_sents]

        assert len(src_sents) == len(trg_sents)
        print("Total sentences:", len(src_sents))

        d = {"src": src_sents, "trg": trg_sents}
        europarl_df = pd.DataFrame(d)
        europarl_df["src_len"] = list(map(lambda x: len(x), europarl_df["src"].str.split(" ")))
        europarl_df["trg_len"] = list(map(lambda x: len(x), europarl_df["trg"].str.split(" ")))
        #### LOWERBOUND ca. 2541134
        europarl_df = europarl_df[(europarl_df.src_len >= 5) | (europarl_df.trg_len >= 5)]
        #### UPPERBOUND ca.  2.336.286
        ## # 10: 225 138, 20: 933.372, 30: 1646686
        europarl_df = europarl_df[(europarl_df.src_len <= 50) & (europarl_df.trg_len <= 50)]
        print(europarl_df.head(10))
        path = os.path.join(BASIC_PREPRO, language_code)

        europarl_df.to_csv(os.path.join(path, DATASET), encoding="utf-8", sep="\t", index=False)

        print("Total duration: {}".format(convert(time.time() - start)))
    else:
        print("File {} already exists!".format(DATASET))



if __name__ == '__main__':
    preprocess_tokenize_europarl_generate_tsv()