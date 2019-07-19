"""

Script for preprocessing raw bilingual corpus files from OPUS

Please download file from the OPUS section: "Statistics and TMX/Moses Downloads", either in txt or tmx format file.
Extract the dataset, put the text or tmx file in a directory and pass this as an argument.

Default path is: data/raw/<corpus_name>/<lang_code>

Ex:

python preprocess.py --lang_code de --type tmx --corpus europarl --max_len 30 --min_len 2 --path data/raw/europarl/de --file de-en.tmx

Conversion:
Converted lines: 1.916.030 (total sentences in the dataset)

Total samples:  1155573
Shuffling data....
Total train: 924458
Total validation: 115557
Total test: 115558

Filtered by length:
Total samples:  1.155.582 (total sentences, with minimum length "min_len" and maximum length "max_len")

"""
import argparse
import os
import re
import time

from project.utils.data.europarl import maybe_download_and_extract_europarl
from project.utils.preprocessing import get_custom_tokenizer, split_data, persist_txt
from project.utils.tmx2corpus.tmx2corpus import Converter, FileOutput
from project.utils.utils import Logger, convert_time_unit
from settings import DATA_DIR_PREPRO, DATA_DIR_RAW


def data_prepro_parser():
    parser = argparse.ArgumentParser(description='Preprocess Europarl Dataset for NMT')
    parser.add_argument("--lang_code", default="de", type=str)
    #  parser.add_argument("--type", default="tmx", type=str, help="TMX")
   # parser.add_argument("--corpus", default="europarl", type=str, help="Corpus name")
    parser.add_argument("--max_len", default=30, type=int, help="Filter sequences with a length <= max_len")
    parser.add_argument("--min_len", default=2, type=int, help="Filter sequences with a length >= min_len")
   # parser.add_argument('--path', default="data/raw/europarl/de", help="Path to raw data files")
   # parser.add_argument('--file', default="de-en.tmx", help="File name after extraction")
    return parser


def raw_preprocess(parser):
    #### preprocessing pipeline for tmx files
    ### download the files #####
    try:
        maybe_download_and_extract_europarl(language_code=parser.lang_code, tmx=True)
    except Exception as e:
        print("An error has occurred:", e)
        print("Please download the parallel corpus manually from: http://opus.nlpl.eu/ | Europarl > Statistics and TMX/Moses Download "
              "\nby selecting the data from the upper-right triangle [en > de]")

    corpus_name = "europarl"
    lang_code = parser.lang_code
    # file_type = parser.type
    path_to_raw_file = os.path.join(DATA_DIR_RAW, corpus_name, lang_code)
    max_len, min_len = parser.max_len, parser.min_len

    file_name = lang_code+"-"+"en"+".tmx"
    COMPLETE_PATH = os.path.join(path_to_raw_file, file_name)

    STORE_PATH = os.path.join(os.path.expanduser(DATA_DIR_PREPRO), corpus_name, lang_code, "splits", str(max_len))
    os.makedirs(STORE_PATH, exist_ok=True)

    start = time.time()
    output_file_path = os.path.join(DATA_DIR_PREPRO, corpus_name, lang_code)

    files = [file for file in os.listdir(output_file_path) if file.startswith("bitext")]
    if "bitext.en" in files and "bitext.{}".format(lang_code) in files:
        print("TMX already converted!")
    else:
        print("Converting tmx to file...")
           ### convert tmx to plain texts - no tokenization is performed
        converter = Converter(output=FileOutput(output_file_path))
        converter.convert([COMPLETE_PATH])
        print("Converted lines:", converter.output_lines)

    target_file = "bitext.{}".format(lang_code)
    src_lines, trg_lines = [], []

    with open(os.path.join(output_file_path, "bitext.en"), 'r', encoding="utf8") as src_file, \
            open(os.path.join(output_file_path, target_file), 'r', encoding="utf8") as target_file:
        for src_line, trg_line in zip(src_file, target_file):
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            if src_line != "" and trg_line != "":
                src_lines.append(src_line)
                trg_lines.append(trg_line)

    ### tokenize lines ####
    assert len(src_lines) == len(trg_lines), "Different lengths!"

    src_tokenizer, trg_tokenizer = get_custom_tokenizer("en", "w", spacy_pretok=False), get_custom_tokenizer("de",
                                                                                                             "w",
                                                                                                             spacy_pretok=False)  # spacy is used
    src_logger = Logger(output_file_path, file_name="bitext.tok.en")
    trg_logger = Logger(output_file_path, file_name="bitext.tok.{}".format(lang_code))

    temp_src_toks, temp_trg_toks = [], []

    with src_tokenizer.nlp.disable_pipes('ner'):
        for i, doc in enumerate(src_tokenizer.nlp.pipe(src_lines, batch_size=1000)):
            tok_doc = ' '.join([tok.text for tok in doc])
            temp_src_toks.append(tok_doc)
            src_logger.log(tok_doc, stdout=True if i % 100000 == 0 else False)

    with trg_tokenizer.nlp.disable_pipes('ner'):
        for i, doc in enumerate(trg_tokenizer.nlp.pipe(trg_lines, batch_size=1000)):
            tok_doc = ' '.join([tok.text for tok in doc])
            temp_trg_toks.append(tok_doc)
            trg_logger.log(tok_doc, stdout=True if i % 100000 == 0 else False)

    if max_len > 0:
        files = ['.'.join(file.split(".")[:2]) for file in os.listdir(STORE_PATH) if
                 file.endswith("tok.en") or file.endswith("tok." + lang_code)]
        filtered_src_lines, filtered_trg_lines = [], []
        if files:
            print("File already reduced by length!")
        else:
            print("Filtering by length...")
            filtered_src_lines, filtered_trg_lines = [], []
            for src_l, trg_l in zip(temp_src_toks, temp_trg_toks):
                ### remove possible duplicate spaces
                src_l_s = re.sub(' +', ' ', src_l)
                trg_l_s = re.sub(' +', ' ', trg_l)
                if src_l_s != "" and trg_l_s != "":
                    src_l_spl, trg_l_spl = src_l_s.split(" "), trg_l_s.split(" ")
                    if len(src_l_spl) <= max_len and len(trg_l_spl) <= max_len:
                        if len(src_l_spl) >= min_len and len(trg_l_spl) >= min_len:
                            filtered_src_lines.append(' '.join(src_l_spl))
                            filtered_trg_lines.append(' '.join(trg_l_spl))

            assert len(filtered_src_lines) == len(filtered_trg_lines)

            src_lines, trg_lines = filtered_src_lines, filtered_trg_lines
            print("Splitting files...")
            train_data, val_data, test_data, samples_data = split_data(src_lines, trg_lines)
            persist_txt(train_data, STORE_PATH, "train.tok", exts=(".en", "." + lang_code))
            persist_txt(val_data, STORE_PATH, "val.tok", exts=(".en", "." + lang_code))
            persist_txt(test_data, STORE_PATH, "test.tok", exts=(".en", "." + lang_code))
           # print("Generating samples files...")
           # persist_txt(samples_data, STORE_PATH, file_name="samples.tok", exts=(".en", "." + lang_code))
    else:

        print("Splitting files...")
        train_data, val_data, test_data, samples_data = split_data(src_lines, trg_lines)
        persist_txt(train_data, STORE_PATH, "train.tok", exts=(".en", "." + lang_code))
        persist_txt(val_data, STORE_PATH, "val.tok", exts=(".en", "." + lang_code))
        persist_txt(test_data, STORE_PATH, "test.tok", exts=(".en", "." + lang_code))
       # print("Generating samples files...")
       # persist_txt(samples_data, STORE_PATH, file_name="samples.tok", exts=(".en", "." + lang_code))

    print("Total time:", convert_time_unit(time.time() - start))


if __name__ == '__main__':
    raw_preprocess(data_prepro_parser().parse_args())
