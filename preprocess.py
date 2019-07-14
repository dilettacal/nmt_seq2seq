"""

Script for preprocessing raw bilingual corpus files from OPUS

Please download file from the OPUS section: "Statistics and TMX/Moses Downloads", either in txt or tmx format file.
Extract the dataset, put the text or tmx file in a directory and pass this as an argument.

Default path is: data/raw/<corpus_name>/<lang_code>

Ex:

python preprocess.py --lang_code de --type tmx --corpus europarl --max_len 30 --min_len 2 --path data/raw/europarl/de --file de-en.tmx

Conversion:
Converted lines: 1.916.030 (total sentences in the dataset)

Filtered by length:
Total samples:  1.148.204 (total sentences, with minimum length "min_len" and maximum length "max_len")

"""
import argparse

from project.utils.preprocessing import raw_preprocess


def data_prepro_parser():
    parser = argparse.ArgumentParser(description='Neural Machine Translation')
    parser.add_argument("--lang_code", default="de", type=str)
    parser.add_argument("--type", default="tmx", type=str, help="TMX")
    parser.add_argument("--corpus", default="europarl", type=str, help="Corpus name")
    parser.add_argument("--max_len", default=30, type=int, help="Filter sequences with a length <= max_len")
    parser.add_argument("--min_len", default=1, type=int, help="Filter sequences with a length >= min_len")
    parser.add_argument('--path', default="data/raw/europarl/de", help="Path to raw data files")
    parser.add_argument('--file', default="de-en.tmx", help="File name after extraction")
    return parser


if __name__ == '__main__':
    raw_preprocess(data_prepro_parser().parse_args())


