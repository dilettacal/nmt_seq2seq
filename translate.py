"""
This script is used to run a trained model on a source file.

Load the model with torch.load, retrieve model information from the checkpoint.
Open the file and process each sentence with the method "predict_from_input".

TODO

"""
import argparse
import os

import torch

from project.model.models import get_nmt_model
from project.utils.arg_parse import str2bool
from project.utils.preprocessing import get_custom_tokenizer
from run_custom_nmt import experiment_parser
from project.utils.constants import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from project.utils.experiment import Experiment
from project.utils.training import predict_from_input
from project.utils.utils import load_embeddings, Logger
from project.utils.vocabulary import get_vocabularies_iterators
from settings import RESULTS_DIR, ROOT, DATA_DIR, BEST_MODEL_PATH

path_to_exp = os.path.join(RESULTS_DIR, "de_en/s/2/uni/2019-07-15-12-13-12/")

path_to_model = os.path.join(path_to_exp, "model.pkl")
print(path_to_model)

# experiment = Experiment(experiment_parser())

experiment = torch.load(os.path.join(path_to_exp, "experiment.pkl"))
print(type(experiment))
print(experiment)

experiment = Experiment(experiment["args"])
print(type(experiment))

train_losses = torch.load(os.path.join(path_to_exp, "train_losses.pkl"))

print((train_losses["values"]))

"""
SRC, TRG, train_iter, val_iter, test_iter, train_data, val_data, test_data, samples, samples_iter = \
        get_vocabularies_iterators(experiment, None)

logger = Logger(path_to_exp, file_name="test.log")
logger.pickle_obj(SRC, "src")
logger.pickle_obj(TRG, "trg")


"""

SRC_loaded = torch.load(os.path.join(path_to_exp, "src.pkl"))
TRG_loaded = torch.load(os.path.join(path_to_exp, "trg.pkl"))

tokens_bos_eos_pad_unk = [TRG_loaded.vocab.stoi[SOS_TOKEN], TRG_loaded.vocab.stoi[EOS_TOKEN],
                          TRG_loaded.vocab.stoi[PAD_TOKEN], TRG_loaded.vocab.stoi[UNK_TOKEN]]

print(len(SRC_loaded.vocab))

print(len(TRG_loaded.vocab))

samples_sentences = open(os.path.join(DATA_DIR, "preprocessed", "europarl", "de", "splits", "30", "samples.tok.de"),
                         encoding="utf-8", mode="r").readlines()
print("Total sentences:", len(samples_sentences))

### loading model

experiment.src_vocab_size = len(SRC_loaded.vocab)
experiment.trg_vocab_size = len(TRG_loaded.vocab)
model = get_nmt_model(experiment, tokens_bos_eos_pad_unk)

model.load_state_dict(torch.load(os.path.join(path_to_exp, "model.pkl")))
model = model.to(experiment.get_device())

print(experiment.__dict__)
logger = Logger(path_to_exp, "live_transl.log")

for sent in samples_sentences:
    predict_from_input(input_sentence=sent, SRC=SRC_loaded, TRG=TRG_loaded, model=model, device=experiment.get_device(),
                       logger=logger)


def translate(root=RESULTS_DIR, path="", predict_from_file=""):
    device = True if torch.cuda.is_available() else False

    if not path:
        path = BEST_MODEL_PATH

    path_to_exp = os.path.join(root, path)
    path_to_model = os.path.join(path_to_exp, "model.pkl")

    experiment = torch.load(os.path.join(path_to_exp, "experiment.pkl"))
    experiment = Experiment(experiment["args"])
    experiment.cuda = device

    SRC_vocab = torch.load(os.path.join(path_to_exp, "src.pkl"))
    TRG_vocab = torch.load(os.path.join(path_to_exp, "trg.pkl"))

    tokens_bos_eos_pad_unk = [TRG_loaded.vocab.stoi[SOS_TOKEN], TRG_loaded.vocab.stoi[EOS_TOKEN],
                              TRG_loaded.vocab.stoi[PAD_TOKEN], TRG_loaded.vocab.stoi[UNK_TOKEN]]

    experiment.src_vocab_size = len(SRC_loaded.vocab)
    experiment.trg_vocab_size = len(TRG_loaded.vocab)
    model = get_nmt_model(experiment, tokens_bos_eos_pad_unk)

    model.load_state_dict(path_to_model)
    model = model.to(device)

    src_tokenizer = get_custom_tokenizer(lang="de", mode="w", fast=False, spacy_pretok=False)

    logger = Logger(path_to_exp, "live_transl.log")

    if predict_from_file != "":

        path_to_file = os.path.join(root, predict_from_file)
        samples = open(path_to_file, encoding="utf-8", mode="r").readlines()
        for sample in samples:
            tok_sample = src_tokenizer.tokenize(sample)
            _ = predict_from_input(input_sentence=tok_sample, SRC=SRC_loaded, TRG=TRG_loaded, model=model,
                               device=experiment.get_device(),
                               logger=logger, stdout=True)
    else:
        input_sequence = ""
        while (1):
            try:
                input_sequence = input("Source > ")
                # Check if it is quit case
                if input_sequence == 'q' or input_sequence == 'quit': break
                input_sequence = src_tokenizer.tokenize(input_sequence.lower())
                out = predict_from_input(model, input_sequence, SRC_vocab, TRG_vocab, logger=logger, device="cuda" if device else "cpu")
                if out:
                    print("Translation > ", out)
                else: print("Error while translating!")

            except KeyError:
                print("Error: Encountered unknown word.")


def translation_parser():
    parser = argparse.ArgumentParser(description='NMT - Neural Machine Translator')

    parser.add_argument('--path', type=str, default="",
                        help='experiment path')
    parser.add_argument('--file', type=str2bool, default="False",
                        help="Translate from keyboard (False) or from samples file (True)")

    args = parser.parse_args()

    translate(".", path=args.path if args.path != "" else None, predict_from_file=args.file)


if __name__ == '__main__':
    parser = translation_parser().parse_args()
    translate(".", path=parser.path, predict_from_file=parser.file)