import datetime
import os
import unittest
import mock
import torch

from project.model.models import get_nmt_model, Seq2Seq
from project.utils.constants import SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN
from project.utils.experiment import Experiment
from project.utils.get_tokenizer import get_custom_tokenizer
from project.utils.tokenizers import SpacyTokenizer, FastTokenizer
from project.utils.utils import Logger
from translate import translate, Translator

user_input = "Die europäische Union ist groß."

class TestTranslation(unittest.TestCase):

    def setUp(self) -> None:
        path_to_model = os.path.expanduser("trained_model")
        use_cuda = False
        device = "cuda" if use_cuda else "cpu"
        MAX_LEN = 30
        path_to_exp = os.path.expanduser(path_to_model)
        path_to_model = os.path.join(path_to_exp, "model.pkl")

        experiment = torch.load(os.path.join(path_to_exp, "experiment.pkl"))
        experiment = Experiment(experiment["args"])
        experiment.cuda = use_cuda

        SRC_vocab = torch.load(os.path.join(path_to_exp, "src.pkl"))
        TRG_vocab = torch.load(os.path.join(path_to_exp, "trg.pkl"))

        src_tokenizer = get_custom_tokenizer(experiment.get_src_lang(), "w", prepro=True)
        trg_tokenizer = get_custom_tokenizer(experiment.get_trg_lang(), "w", prepro=True)

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
        logger.log("Beam width: {}".format(5))

        self.translator = Translator(model, SRC_vocab, TRG_vocab, logger,
                                     src_tokenizer, trg_tokenizer, device, beam_size=1,
                                max_len=MAX_LEN)


    def test_load_data_for_translation(self):
        out = self.translator.predict_sentence(user_input)
        self.assertIsInstance(out, str)

    def test_translator(self):
        self.assertIsInstance(self.translator, Translator)
        try:
            self.assertIsInstance(self.translator.src_tokenizer, FastTokenizer)
        except AssertionError:
            self.assertIsInstance(self.translator.src_tokenizer, SpacyTokenizer)

        try:
            self.assertIsInstance(self.translator.trg_tokenizer, FastTokenizer)
        except AssertionError:
            self.assertIsInstance(self.translator.trg_tokenizer, SpacyTokenizer)

        self.assertIsInstance(self.translator.model, Seq2Seq)
        self.assertEqual(self.translator.device, "cpu")
        self.assertEqual(self.translator.max_len, 30)
        self.assertEqual(self.translator.get_beam_size(), 1)

    def test_change_beam_size(self):
        self.assertEqual(self.translator.get_beam_size(), 1)
        self.translator.set_beam_size(4)
        self.assertEqual(self.translator.get_beam_size(), 4)




