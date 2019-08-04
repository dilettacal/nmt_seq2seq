"""
This script is used to run a trained model on a source file.

Load the model with torch.load, retrieve model information from the checkpoint.
Open the file and process each sentence with the method "predict_from_input".

"""
import argparse
import codecs
import datetime
import io
import os
import readline
import sys

import torch

from project.model.models import get_nmt_model
from project.utils.get_tokenizer import get_custom_tokenizer
from project.utils.constants import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from project.utils.experiment import Experiment
from project.utils.training import predict_from_input, check_translation
from project.utils.utils import Logger

UTF8Reader = codecs.getreader('utf8')
sys.stdin = UTF8Reader(sys.stdin)

class Translator(object):
    def __init__(self, model, SRC, TRG, logger, src_tokenizer, trg_tokenizer, device="cuda", beam_size=5, max_len=30):
        self.model = model
        self.src_vocab = SRC
        self.trg_vocab = TRG
        self.logger = logger
        self.device = device
        self.beam_size = beam_size
        self.max_len = max_len
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer


    def predict_sentence(self, sentence, stdout=False):

        sentence = self.src_tokenizer.tokenize(sentence.lower())
        #### Changed from original ###
        sent_indices = [self.src_vocab.vocab.stoi[word] if word in self.src_vocab.vocab.stoi
                        else self.src_vocab.vocab.stoi[UNK_TOKEN] for word in
                        sentence]
        sent = torch.LongTensor([sent_indices])
        sent = sent.to(self.device)
        sent = sent.view(-1, 1)  # reshape to sl x bs
        self.logger.log('SRC  >>> ' + ' '.join([self.src_vocab.vocab.itos[index] for index in sent_indices]),
                        stdout=stdout)
        ### predict sentences with beam search 5
        pred = self.model.predict(sent, beam_size=self.beam_size, max_len=self.max_len)
        pred = [index for index in pred if index not in [self.trg_vocab.vocab.stoi[SOS_TOKEN],
                                                         self.trg_vocab.vocab.stoi[EOS_TOKEN]]]
        out = ' '.join(self.trg_vocab.vocab.itos[idx] for idx in pred)
        self.logger.log('PRED >>> ' + out, stdout=True)
        return out

    def predict_from_text(self, path_to_file):
        path_to_file = os.path.expanduser(path_to_file)
        self.logger.log("Predictions from file: {}".format(path_to_file))
        self.logger.log("-" * 100, stdout=True)
        # read file
        with open(path_to_file, encoding="utf-8", mode="r") as f:
            samples = f.readlines()
        samples = [x.strip().lower() for x in samples if x]
        for sample in samples:
            out = self.predict_sentence(sample)
            self.logger.log("-" * 100, stdout=True)

    def set_beam_size(self, new_size):
        self.beam_size = new_size

    def get_beam_size(self):
        return self.beam_size

def translate(path="", predict_from_file="", beam_size=5):
    use_cuda = True if torch.cuda.is_available() else False
    device = "cuda" if use_cuda else "cpu"
    FIXED_WORD_LEVEL_LEN = 30

    if not path:
        print("Please provide path to model!")
        return False

    path_to_exp = os.path.expanduser(path)
    print("Using experiment from: ", path_to_exp)
    path_to_model = os.path.join(path_to_exp, "model.pkl")

    try:
        experiment = torch.load(os.path.join(path_to_exp, "experiment.pkl"))
        experiment = Experiment(experiment["args"])
        experiment.cuda = use_cuda
    except FileNotFoundError as e:
        print("Wrong path. File not found: ", e)
        return



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
    try:
        model.load_state_dict(torch.load(path_to_model))
    except FileNotFoundError as e:
        print("Wrong path. File not found: ", e)
        return
    model = model.to(device)

    logger = Logger(path_to_exp, "live_transl.log")
    logger.log("Live translation: {}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), stdout=False)
    logger.log("Beam width: {}".format(beam_size))

    translator = Translator(model, SRC_vocab, TRG_vocab, logger, src_tokenizer, trg_tokenizer, device, beam_size, max_len=MAX_LEN)

    if predict_from_file:
        translator.predict_from_text(predict_from_file)
    else:
        input_sequence = ""
        while (1):
            try:
                try:
                    input_sequence = input("SRC  >>> ")
                    if input_sequence.lower().startswith("#"):
                        bs = input_sequence.split("#")[1]
                        try:
                            beam_size = int(bs)
                            logger.log("New Beam width: {}".format(beam_size))
                            input_sequence = input("SRC  >>> ")
                        except ValueError:
                            input_sequence = input("SRC  >>> ")
                except ValueError as e:
                    print("An error has occurred: {}. Please restart program!".format(e))
                    return False
                # Check if it is quit case
                if input_sequence == 'q' or input_sequence == 'quit': break
                translator.set_beam_size(beam_size)
                out = translator.predict_sentence(input_sequence)
                if out:
                    logger.log("-"*35, stdout=True)
                else: print("Error while translating!")

            except KeyError:
                print("Error: Encountered unknown word.")

    return [experiment, model, SRC_vocab, TRG_vocab, src_tokenizer, trg_tokenizer, logger]

def translation_parser():
    parser = argparse.ArgumentParser(description='NMT - Neural Machine Translator')

    parser.add_argument('--path', type=str, default="",
                        help='experiment path')
    parser.add_argument('--file', type=str, default="",
                        help="Translate from keyboard (False) or from samples file (True)")
    parser.add_argument('--beam', type=int, default=5, help="beam size")
    return parser


if __name__ == '__main__':
    parser = translation_parser().parse_args()
    #parser.path = BEST_BASELINE_TIED
    _ = translate(path=parser.path, predict_from_file=parser.file, beam_size = parser.beam)