import argparse

from project.utils.utils import str2number


def data_prepro_parser():
    parser = argparse.ArgumentParser(
        description='Preprocess Europarl Dataset for NMT. \nThis script allows you to preprocess and tokenize the Europarl Dataset.')
    parser.add_argument("--lang_code", default="de", type=str,
                        help="First language is English. Specifiy with 'lang_code' the second language as language code (e.g. 'de').")
    parser.add_argument("--test_ratio", help="Specify the test ratio. Standard: 3000 samples. If you pass a float, this will be split the data based on that proportion, e.g. 0.1 --> 0.8, 0.1, 0.1",
                        type=str2number, default=3000)
    return parser