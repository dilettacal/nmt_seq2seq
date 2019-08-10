from argparse import Namespace
import torch

class Experiment(object):
    """
    The Experiment class defines the configuration for a training experiment.
    """
    SEQ_MAX_LEN = 30
    def __init__(self, parser):
        self.model_type = 'none' # setup in the run_custom_nmt script
        if isinstance(parser, Namespace):
            self.args = parser
        else: self.args = parser.parse_args()
        self.epochs = self.args.epochs
        self.batch_size = self.args.b
        self.voc_limit = self.args.v
        self.corpus = self.args.corpus
        self.lang_code = self.args.lang_code
        self.reverse_lang_comb = self.args.reverse
        self.min_freq = self.args.min if self.args.min >= 0 else 5
        self.tied = self.args.tied

        assert self.args.attn in ["none", "dot"]
        self.attn = self.args.attn
        self.bi = self.args.bi
        self.reverse_input = self.args.reverse_input
        if self.bi and self.reverse_input:
            self.bi = True
            self.reverse_input = False
        elif self.reverse_input and not self.bi:
            self.reverse_input = True
       # self.reverse_input = True if not self.bi else False
        if self.args.max_len > Experiment.SEQ_MAX_LEN:
            self.args.max_len = Experiment.SEQ_MAX_LEN
        self.truncate = self.args.max_len
        self.data_dir = self.args.data_dir

        self.src_lang = self.lang_code if self.reverse_lang_comb == True else "en"
        self.trg_lang = self.lang_code if self.src_lang == "en" else "en"

        self.cuda = self.args.cuda
        self.lr = self.args.lr

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
        self.nlayers = self.args.num_layers
        self.dp = self.args.dp
        self.tok = self.args.tok
        self.val_beam_size = self.args.beam

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

    def check_norm(self):
        return self.args.norm