"""
This script is used to run a trained model on a source file.

Load the model with torch.load, retrieve model information from the checkpoint.
Open the file and process each sentence with the method "predict_from_input".

"""
import codecs
import datetime
import os
import sys

import torch

from project.model.models import get_nmt_model
from project.utils.get_tokenizer import get_custom_tokenizer
from project.utils.constants import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from project.utils.experiment import Experiment
from project.utils.parsers.get_translation_parser import translation_parser
from project.utils.train_preprocessing import get_vocabularies_iterators
from project.utils.translators import Translator
from project.utils.utils import Logger

UTF8Reader = codecs.getreader('utf8')
sys.stdin = UTF8Reader(sys.stdin)


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
        return False

    logger_file_name = experiment.rnn_type+"_live_translations.log"
    logger = Logger(path_to_exp,file_name=logger_file_name)

    try:
        SRC_vocab = torch.load(os.path.join(path_to_exp, "src.pkl"))
        TRG_vocab = torch.load(os.path.join(path_to_exp, "trg.pkl"))
    except ModuleNotFoundError as e:
        print("Error while loading vocabularies: {}\nLoading vocabularies based on experiment configuration...".format(e))
        train_prepos = get_vocabularies_iterators(experiment)
        SRC_vocab, TRG_vocab = train_prepos[0], train_prepos[1]
        logger.pickle_obj(SRC_vocab, "src")
        logger.pickle_obj(TRG_vocab, "trg")

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

    logger.log("Live translation: {}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), stdout=False)
    logger.log("Beam width: {}".format(beam_size))

    translator = Translator(model, SRC_vocab, TRG_vocab, logger, src_tokenizer, device, beam_size, max_len=MAX_LEN)

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


if __name__ == '__main__':
    parser = translation_parser().parse_args()
    #parser.path = BEST_BASELINE_TIED
    _ = translate(path=parser.path, predict_from_file=parser.file, beam_size = parser.beam)