import os
import string
from datetime import datetime

import numpy as np
import pandas as pd
import unidecode as unidecode
from sacremoses import MosesTokenizer
import unicodedata

import re

from project import get_full_path
from project.utils.data.mappings import ENG_CONTRACTIONS_MAP, UMLAUT_MAP
from project.utils.utils import Logger
from settings import RAW_EUROPARL, DATA_DIR_PREPRO


def expand_contraction(sentence, mapping):
    """
    Expands tokens in sentence given a contraction dictionary

    See: https://www.linkedin.com/pulse/processing-normalizing-text-data-saurav-ghosh

    :param sentence: sentence to expand
    :param mapping: contraction dictionary
    :return: expanded sentence
    """
    contractions_patterns = re.compile('({})'.format('|'.join(mapping.keys())), flags=re.IGNORECASE | re.DOTALL)

    def replace_text(t):
        txt = t.group(0)
        if txt.lower() in mapping.keys():
            return mapping[txt.lower()]

    expanded_sentence = contractions_patterns.sub(replace_text, sentence)
    return expanded_sentence


def char_filter(string):
    valid = re.compile('[a-zA-Z0-9]+')
    for char in unicodedata.normalize('NFC', string):
        decoded = unidecode.unidecode(char)
        if valid.match(decoded):
            yield char
        else:
            yield decoded

def clean_string(string):
    return "".join(char_filter(string))


def clearup(s, chars, replacee):
    s = re.sub('[%s]' % chars, replacee, s)
    if replacee == '':
        s = re.sub(' +', ' ', s)
    return s


def basic_preprocess_sentence(sent, lang):
   # print("Raw sentence:", sent)
    copy = str(sent)+""

    if lang == "en":
        sent = expand_contraction(sent, ENG_CONTRACTIONS_MAP)
    elif lang == "de":
        sent = expand_contraction(sent, UMLAUT_MAP)
    ### Regex ###
    space_before_mark = r"\s+([.?!])"

    before_apos = r"\s+(['])"
    after_apos = r"(['])\s+([\w])"
    ## remove extra spaces --> the normalize function would output 'house \?' instead of 'house ?"
    sent = re.sub(space_before_mark, r"\1", sent)

        ## Remove hyphens
    sent = sent.replace('-', '')
    ### remove extra spaces before apostroph, if any
    sent = re.sub(before_apos, r"\1", sent)
    sent = re.sub(after_apos, r"\1\2", sent)

    tokenizer = MosesTokenizer(lang)
    sent = tokenizer.tokenize(sent.strip())

    sent = ' '.join(sent)
    sent = clean_string(sent)

    sent = clearup(sent, string.digits, "*")
    sent = re.sub('\*+', 'NUM', sent)

    sent = clearup(sent, string.punctuation, '')
    sent = sent.strip().split(" ")
    sent = [word if word.isupper() else word.lower() for word in sent]
    sent = ' '.join(sent)

    if not sent or sent == "":
        sent = ""
    return sent



def preprocess_corpus(src_sents, trg_sents, language_code, max_len=30):
    for src_sent, trg_sent in zip(src_sents, trg_sents):
        src_sent = basic_preprocess_sentence(src_sent, "en")
        trg_sent = basic_preprocess_sentence(trg_sent, language_code)
        if src_sent and trg_sent:
            src_splits = src_sent.split(" ")
            trg_splits = trg_sent.split(" ")
            if (len(src_splits) <= max_len and len(trg_splits) <= max_len):
                yield (src_sent, trg_sent)


flatten = lambda l: [item for sublist in l for item in sublist]


def convert_txt_to_tsv(root_path, src_data, trg_data):

    raw_data = {'src': [line for line in src_data], 'trg': [line for line in trg_data]}

    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    df['src_len'] = df['src'].str.count(' ')
    df['trg_len'] = df['trg'].str.count(' ')
  #  df = df.query('src_len < 80 & trg_len < 80')
   # df = df.query('src_len < trg_len * 1.5 & src_len * 1.5 > trg_len')

    df.to_csv(os.path.join(root_path, RAW_EUROPARL), sep="\t", encoding="utf-8", index=False)
    print("Total sentences:", len(df))
    print(pd.read_csv(os.path.join(root_path, RAW_EUROPARL), sep="\t").head())

def read_from_tsv(path):
    return pd.read_csv(path, encoding="utf-8", sep="\t")


def tokenize_input(text, lang ="en", char_level=False, reversed=False):
    """
    Tokenization method
    :param text:
    :param lang:
    :param char_level:
    :param reversed:
    :return:
    """

    if not char_level:
        tokenized = basic_preprocess_sentence(text, lang=lang)
    else: tokenized = list(text)
    if reversed:
        tokenized = tokenized[::-1]
    return tokenized


def generate_splits_from_datasets(max_len, root=os.path.join(DATA_DIR_PREPRO, "europarl"), language_code="de",
                                  filename="europarl.tsv", save_as="tsv"):
    """
    Generate train, val and test splits based on the ratio and the given filter query
    :param path_to_dataset: complete path to the .tsv file
    :param max_len: maximum sentence lenghts
    :param val_ratio:
    :param language_code:
    :return:
    """

    path_to_dataset = os.path.join(root, language_code, filename)

    if not os.path.isfile(path_to_dataset):
        print("Missing dataset to load from: {}".format('.'.join(path_to_dataset.split(".")[:-1])))
        exit(-1)

    store_path = os.path.join(DATA_DIR_PREPRO, "europarl", language_code, str(max_len))
    os.makedirs(store_path, exist_ok=True)

    files = os.listdir(store_path)
    assert save_as == "tsv" or save_as == "txt", "Wrong format!"

    if save_as == "tsv":
        if "train.tsv" in files and "val.tsv" in files and "test.tsv" in files:
            print("Splits for max sequence length {} already preprocessed!".format(max_len))
            return
    else:
        FILES = sorted(["train.en", "val.en", "test.en",
                        "train.{}".format(language_code), "val.{}".format(language_code),
                        "test.{}".format(language_code)])

        files = sorted([file.lower() for file in files if
                        (file.endswith(".en") or file.endswith(".{}".format(language_code))) and (
                                file.split(".")[0] in ["train", "test", "val"])])
        if files == FILES:
            print("Splits for max sequence  length {} already preprocessed".format(max_len))
            return

    ### Define data logger ###
    split_logger = Logger(store_path, file_name="split.log")
    split_logger.log("Splitting information - {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


    main_df = read_from_tsv(path_to_dataset)
    split_logger.log("Original dataset contains {} samples.".format(len(main_df)))
    main_df = main_df[(main_df.src_len <= max_len) & (main_df.trg_len <= max_len)]
    split_logger.log("Dataset samples have been reduced to {} examples.".format(len(main_df)))

    if len(main_df) >= 200000:
        fractions = np.array([0.6, 0.2, 0.2])
    else:
        fractions = np.array([0.7, 0.2, 0.1])

    # shuffle your input
    df_to_split = main_df.sample(frac=1)
    # split into 3 parts
    train, val, test = np.array_split(df_to_split, (fractions[:-1].cumsum() * len(df_to_split)).astype(int))

    print("Train samples:", train[:5], sep="\n")
    print("Val samples:", val[:5], sep="\n")
    print("Test samples:", test[:5], sep="\n")
    print("")

    if save_as == "tsv":

        train.to_csv(os.path.join(store_path, "train.tsv"), encoding="utf-8", sep="\t", index=False)
        val.to_csv(os.path.join(store_path, "val.tsv"), encoding="utf-8", sep="\t", index=False)
        test.to_csv(os.path.join(store_path, "test.tsv"), encoding="utf-8", sep="\t", index=False)

    elif save_as == "txt":
        pass

    else: raise Exception("Format not supported!")

    split_logger.log("Train samples: {}".format(len(train)))
    split_logger.log("Validation samples: {}".format(len(val)))
    split_logger.log("Testing samples: {}".format(len(test)))
    print("Splits created!")


def generate_splits_from_plain_text(root=os.path.join(DATA_DIR_PREPRO, "europarl"), language_code="de", max_len=30, filename="europarl"):
    FILES = sorted(["train.en", "val.en", "test.en",
             "train.{}".format(language_code), "val.{}".format(language_code), "test.{}".format(language_code)])

    store_path = os.path.join(DATA_DIR_PREPRO, "europarl", language_code, str(max_len))
    os.makedirs(store_path, exist_ok=True)
    files = os.listdir(store_path)
    files = sorted([file.lower() for file in files if
                    (file.endswith(".en") or file.endswith(".{}".format(language_code))) and (
                            file.split(".")[0] in ["train", "test", "val"])])

    if files == FILES:
        print("Files already splitted!")
    else:
        split_logger = Logger(store_path, file_name="split.log")
        split_logger.log("Splitting information - {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        en_file = filename+".en"
        trg_file = filename+".{}".format(language_code)
        path_to_en = os.path.join(root, language_code, en_file)
        path_to_de = os.path.join(root, language_code, trg_file)



if __name__ == '__main__':
    language_code = "de"
    FILES = sorted(["train.en", "val.en", "test.en",
             "train.{}".format(language_code), "val.{}".format(language_code), "test.{}".format(language_code)])

    files = ["train.en", "val.en", "test.en",
                    "train.de", "val.de", "test.de", "europarl.de", "bitext.de"]



    print(files)
    print(FILES == files)