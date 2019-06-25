import argparse

import torch

from settings import VALID_MODELS, VALID_DEC
from project.utils.utils import Logger


def str2bool(v):
    #https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2float(s):
    try:
        return float(s)
    except ValueError:
        return None

def str2array(s):
    if s:
        s = s.split(" ")
    return s

def experiment_parser():
    parser = argparse.ArgumentParser(description='Neural Machine Translation')
    parser.add_argument('--lr', default=2e-3, type=float, metavar='N', help='learning rate, default: 2e-3')
    parser.add_argument('--hs', default=500, type=int, metavar='N', help='size of hidden state, default: 300')
    parser.add_argument('--emb', default=500, type=int, metavar='N', help='embedding size, default: 300')
    parser.add_argument('--nlayers', default=4, type=int, metavar='N', help='number of layers in rnn, default: 2')
    parser.add_argument('--dp', default=0.25, type=float, metavar='N', help='dropout probability, default: 0.30')
    parser.add_argument('--bi', type=str2bool, default=False,
                        help='use bidrectional encoder, default: false')
    parser.add_argument('--reverse_input', type=str2bool, default=False,
                        help='reverse input to encoder, default: False')
    parser.add_argument('-v', default=30000, type=int, metavar='N', help='vocab size, use 0 for maximum size, default: 0')
    parser.add_argument('-b', default=64, type=int, metavar='N', help='batch size, default: 64')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of epochs, default: 50')
    parser.add_argument('--model', metavar='DIR', default=None, help='path to model, default: None')

    parser.add_argument('--predict_from_input', metavar='STR', default=None, help='Source sentence to translate')
    parser.add_argument('--max_len', type=int, metavar="N", default=30, help="Sequence max length. Default 30 units.")
    parser.add_argument('--model_type', default="custom", metavar='STR', help="Model type (custom, cho, sutskever)")
    parser.add_argument('--dec', type=str, default="standard", help="Decoder type: Standard, Context, attn")

    parser.add_argument('--corpus', default="europarl", metavar='STR',
                        help="The corpus, where training should be performed. Possible values: \'europarl\' and \'simple'\ - the iwslt dataset from torchtext")

    parser.add_argument('-c', metavar='STR', default=False, help="Training at char level")

    parser.add_argument('-lang_code', metavar='STR', default="de",
                        help="Provide language code, e.g. 'de'. This is the source or target language.")

    parser.add_argument('--reverse', metavar="STR", default=False,
                        help="Reverse language combination. Standard: EN > <lang>, if reverse, then <lang> > EN")

    parser.add_argument('--cuda', type=str2bool, default="True")

    parser.add_argument('--rnn', metavar="STR", default="lstm")
    parser.add_argument('--maxout', type=int, default=0, help="Maxout units")

    parser.add_argument('--train', default=200000, type=int, help="Number of training examples")
    parser.add_argument('--val', default=20000, type=int, help="Number of validation examples")
    parser.add_argument('--test', default=10000, type=int, help="Number of test examples")

    return parser


class Experiment(object):
    def __init__(self):
        self.args = experiment_parser().parse_args()

        assert self.args.model_type.lower() in VALID_MODELS
        assert self.args.dec in VALID_DEC
        #### Training configurations
        self.epochs = self.args.epochs
        self.batch_size = self.args.b
        self.voc_limit = self.args.v
        self.corpus = self.args.corpus
        self.lang_code = self.args.lang_code
        self.reverse_lang_comb = self.args.reverse
        self.model_type = self.args.model_type

        self.src_lang = self.lang_code if self.reverse_lang_comb else "en"
        self.trg_lang = "en" if self.src_lang == self.lang_code else self.lang_code

        self.cuda = self.args.cuda
        self.lr = self.args.lr
        self.reverse_input = self.args.reverse_input
        self.char_level = self.args.c
        self.max_len = self.args.max_len

        self.src_vocab_size = None
        self.trg_vocab_size = None


        ### samples config
        self.train_samples = self.args.train
        self.val_samples = self.args.val
        self.test_samples = self.args.test

        self.reduce = [self.train_samples, self.val_samples, self.test_samples]

        #### Model configurations
        self.hid_dim = self.args.hs
        self.emb_size = self.args.emb
        self.rnn_type = self.args.rnn
        self.nlayers = self.args.nlayers
        self.dp = self.args.dp
        self.bi = self.args.bi

        self.decoder_type = self.args.dec
        if self.decoder_type == "context":
            self.rnn_type = "gru"


    def get_args(self):
        return self.args

    def get_src_lang(self):
        return self.src_lang

    def get_trg_lang(self):
        return self.trg_lang

    def get_device(self):
        return torch.device("cuda") if (self.cuda and torch.cuda.is_available()) else torch.device("cpu")

    def get_dict(self):
        return self.__dict__



