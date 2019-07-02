"""

Script for preprocessing raw bilingual corpus files from OPUS

Please download file from the OPUS section: "Statistics and TMX/Moses Downloads", either in txt or tmx format file.
Extract the dataset, put the text or tmx file in a directory and pass this as an argument.

Default path is: data/raw/<corpus_name>/<lang_code>

Ex:

python preprocess.py --lang_code de --type tmx --corpus europarl --max_len 30 --min_len 2 --path data/raw/europarl/de --file de-en.tmx

"""
import argparse
import os
import time

from tmx2corpus import FileOutput

from project.experiment.setup_experiment import str2bool
from project.utils.data.preprocessing import TMXConverter, get_custom_tokenizer
from project.utils.utils import convert
from settings import DATA_DIR_PREPRO, DATA_DIR_RAW


def data_prepro_parser():
    parser = argparse.ArgumentParser(description='Neural Machine Translation')
    parser.add_argument("--lang_code", default="de", type=str)
    parser.add_argument("--type", default="tmx", type=str, help="TMX or TXT")
    parser.add_argument("--corpus", default="europarl", type=str, help="Corpus name")
    parser.add_argument("--max_len", default=30, type=int, help="Filter sequences with a length <= max_len")
    parser.add_argument("--min_len", default=1, type=int, help="Filter sequences with a length >= min_len")
    parser.add_argument('--path', default="data/raw/europarl/de", help="Path to raw data files")
    parser.add_argument('--file', default="de-en.tmx", help="File name after extraction")
    parser.add_argument('--v', type=str2bool, default="True", help="Either vocabulary should be reduced by replacing some repeating tokens with labels.\nNumbers are replaced with NUM, Persons names are replaced with PERSON. Require: Spacy!")

    return parser



if __name__ == '__main__':
    #### preprocessing pipeline for tmx files

    parser = data_prepro_parser().parse_args()
    corpus_name = parser.corpus
    lang_code = parser.lang_code
    file_type = parser.type
    path_to_raw_file = parser.path
    reduce_vocab = parser.v
    max_len, min_len = parser.max_len, parser.min_len

    COMPLETE_PATH = os.path.join(path_to_raw_file, parser.file)

    assert file_type in ["tmx", "txt"]

    if file_type == "tmx":
        start = time.time()
        FILE = os.path.join(DATA_DIR_RAW, corpus_name, lang_code)
        output_file_path = os.path.join(DATA_DIR_PREPRO, corpus_name, lang_code)
        converter = TMXConverter(output=FileOutput(output_file_path))
        tokenizers = [get_custom_tokenizer("", "w", "fast"), get_custom_tokenizer("", "w", "fast")]
        converter.add_tokenizers(tokenizers)
        converter.convert([COMPLETE_PATH]) #---> bitext.en, bitext.de, bitext.tok.de, bitext.tok.en
        print("Total time:", convert(time.time() - start))
        print(converter.output_lines)

        src_lines = [line.strip("\n") for line in
                     open(os.path.join(output_file_path, "bitext.tok.en"), mode="r",
                          encoding="utf-8").readlines() if line]
        trg_lines = [line.strip("\n") for line in
                     open(os.path.join(output_file_path, "bitext.tok.de"), mode="r",
                          encoding="utf-8").readlines() if line]

        #### tokenize with spacy if available

        if max_len > 0 or min_len > 0:
            filtered_src_lines, filtered_trg_lines = [], []
            for src_l, trg_l in zip(src_lines, trg_lines):
                if src_l != "" and trg_l != "":
                    src_l_s, trg_l_s = src_l.split(" "), trg_l.split(" ")
                    if (len(src_l_s) <= max_len and len(src_l_s) >= min_len) and (len(trg_l_s) <= max_len and len(trg_l_s) >= min_len):
                        filtered_trg_lines.append(' '.join(src_l_s))
                        filtered_trg_lines.append(' '.join(trg_l_s))

        if reduce_vocab:
            #### reduce vocabulary by replacing some particular tokens
            pass
        else:
            ### only tokenize with spacy if available
            pass
    else:
        #TODO
        pass



    splits = [80,20,10]


    train_data, val_data, test_data = split_data(src_lines, trg_lines)
    print("Samples:")
    print(len(train_data[0]), len(val_data[0]), len(test_data[0]))
    store = os.path.expanduser(DATA_DIR_PREPRO)
    store = os.path.join(store, "europarl", "de", "splits")
    # store_path, file_name, exts
    persist_txt(train_data, store, "train", exts=(".en", ".de"))
    persist_txt(val_data, store, "val", exts=(".en", ".de"))
    persist_txt(test_data, store, "test", exts=(".en", ".de"))

