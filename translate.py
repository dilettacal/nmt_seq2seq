"""
This script is used to run a trained model on a source file.

Load the model with torch.load, retrieve model information from the checkpoint.
Open the file and process each sentence with the method "predict_from_input".

"""
import argparse
import datetime
import os

import torch

from project.model.models import get_nmt_model
from project.utils.tokenizers import get_custom_tokenizer
from project.utils.constants import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from project.utils.experiment import Experiment
from project.utils.training import predict_from_input
from project.utils.utils import Logger, str2bool


def translate(path="", predict_from_file="", beam_size=5):
    use_cuda = True if torch.cuda.is_available() else False
    device = "cuda" if use_cuda else "cpu"
    FIXED_WORD_LEVEL_LEN = 30

    if not path:
        print("Please provide path to model!")
        return

    path_to_exp = os.path.expanduser(path)
    print("Using experiment from: ", path_to_exp)
    path_to_model = os.path.join(path_to_exp, "model.pkl")

    experiment = torch.load(os.path.join(path_to_exp, "experiment.pkl"))
    experiment = Experiment(experiment["args"])
    experiment.cuda = use_cuda

    SRC_vocab = torch.load(os.path.join(path_to_exp, "src.pkl"))
    TRG_vocab = torch.load(os.path.join(path_to_exp, "trg.pkl"))

    tok_level = "w"

    src_tokenizer = get_custom_tokenizer(experiment.get_src_lang(), "w", prepro=True)
    trg_tokenizer = get_custom_tokenizer(experiment.get_trg_lang(), "w", prepro=True)
    MAX_LEN = FIXED_WORD_LEVEL_LEN

    SRC_vocab.tokenize = src_tokenizer.tokenize
    TRG_vocab.tokenize = trg_tokenizer.tokenize

    tokens_bos_eos_pad_unk = [TRG_vocab.vocab.stoi[SOS_TOKEN], TRG_vocab.vocab.stoi[EOS_TOKEN],
                              TRG_vocab.vocab.stoi[PAD_TOKEN], TRG_vocab.vocab.stoi[UNK_TOKEN]]

    experiment.src_vocab_size = len(SRC_vocab.vocab)
    experiment.trg_vocab_size = len(TRG_vocab.vocab)
    model = get_nmt_model(experiment, tokens_bos_eos_pad_unk)
    model.load_state_dict(torch.load(path_to_model))
    model = model.to(device)

    logger = Logger(path_to_exp, "live_transl.log")
    logger.log("Live translation: {}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), stdout=False)

    if predict_from_file:

        path_to_file = os.path.expanduser(predict_from_file)
        samples = open(path_to_file, encoding="utf-8", mode="r").readlines()
        logger.log("Predictions from file: {}".format(path_to_file))
        for sample in samples:
            tok_sample = src_tokenizer.tokenize(sample)
            _ = predict_from_input(input_sentence=tok_sample, SRC=SRC_vocab, TRG=TRG_vocab, model=model,
                               device=experiment.get_device(),
                               logger=logger, stdout=True, beam_size=beam_size, max_len=MAX_LEN)
    else:
        input_sequence = ""
        while (1):
            try:
                input_sequence = input("Source > ")
                # Check if it is quit case
                if input_sequence == 'q' or input_sequence == 'quit': break
                input_sequence = src_tokenizer.tokenize(input_sequence.lower())
                out = predict_from_input(model, input_sequence, SRC_vocab, TRG_vocab, logger=logger, device="cuda" if use_cuda else "cpu",
                                         beam_size=beam_size, max_len=MAX_LEN)
                if out:
                    print("Translation > ", out)
                else: print("Error while translating!")

            except KeyError:
                print("Error: Encountered unknown word.")


def translation_parser():
    parser = argparse.ArgumentParser(description='NMT - Neural Machine Translator')

    parser.add_argument('--path', type=str, default="",
                        help='experiment path')
    parser.add_argument('--file', type=str, default="",
                        help="Translate from keyboard (False) or from samples file (True)")
    parser.add_argument('--beam', type=int, default=5, help="beam size")
    return parser


BEST_BASELINE = "best_baseline/de_en/s/2/uni/2019-07-15-14-03-09/"
BEST_BASELINE_TIED = "best_baseline_tied/de_en/s/2/uni/2019-07-15-14-03-30"

if __name__ == '__main__':
    parser = translation_parser().parse_args()
    #parser.path = BEST_BASELINE_TIED
    translate(path=parser.path, predict_from_file=parser.file, beam_size = parser.beam)