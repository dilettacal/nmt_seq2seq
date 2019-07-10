import torch

from settings import VALID_MODELS


class Experiment(object):
    def __init__(self, parser):
        # self.args = experiment_parser().parse_args()
        self.args = parser.parse_args()

        assert self.args.model_type.lower() in VALID_MODELS
        #### Training configurations
        self.epochs = self.args.epochs
        self.batch_size = self.args.b
        self.voc_limit = self.args.v
        self.corpus = self.args.corpus
        self.lang_code = self.args.lang_code
        self.reverse_lang_comb = self.args.reverse
        # print("Reverse?", self.reverse_lang_comb)
        self.model_type = self.args.model_type
        self.min_freq = self.args.min if self.args.min >= 0 else 5
        self.tied = self.args.tied
        self.pretrained = self.args.pretrained

        assert self.args.attn in ["none", "additive", "dot"]
        self.attn = self.args.attn

        self.bi = self.args.bi
        if self.model_type == "s":
            self.reverse_input = True
            self.bi = False
        else:
            self.reverse_input = self.args.reverse_input

        self.truncate = self.args.max_len
        self.data_dir = self.args.data_dir

        self.src_lang = self.lang_code if self.reverse_lang_comb == True else "en"
        self.trg_lang = self.lang_code if self.src_lang == "en" else "en"

        self.cuda = self.args.cuda
        self.lr = self.args.lr

        self.char_level = self.args.c

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

        self.tok = self.args.tok

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

    def set_train(self, n):
        self.train_samples = n

    def set_val(self, n):
        self.val_samples = n

    def set_test(self, n):
        self.test_samples = n