import argparse


def translation_parser():
    parser = argparse.ArgumentParser(description='NMT - Neural Machine Translator')
    parser.add_argument('--path', type=str, default="",
                        help='experiment path. Provide this as relative path e.g.: results/final_local/custom/lstm/2/bi/2019-08-04-04-53-04')
    parser.add_argument('--file', type=str, default="",
                        help="Translate from file. Please provide path to file e.g. ./translations.txt ")
    parser.add_argument('--beam', type=int, default=5, help="Model beam size.")
    return parser