import argparse


def data_prepro_parser():
    parser = argparse.ArgumentParser(
        description='Preprocess Europarl Dataset for NMT. \nThis script allows you to preprocess and tokenize the Europarl Dataset.')
    parser.add_argument("--lang_code", default="de", type=str,
                        help="First language is English. Specifiy with 'lang_code' the second language as language code (e.g. 'de').")
    return parser