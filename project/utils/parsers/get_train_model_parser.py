import argparse

from project.utils.utils_functions import str2bool


def experiment_parser():
    """
    Experiment parser parses arguments for the experiment
    :return:
    """
    parser = argparse.ArgumentParser(description='Neural Machine Translation with PyTorch')
    parser.add_argument('--lr', default=2e-4, type=float, metavar='N', help='Learning rate, default: 0.0002')
    parser.add_argument('--hs', default=300, type=int, metavar='N', help='Size of hidden state, default: 300')
    parser.add_argument('--emb', default=300, type=int, metavar='N', help='Embedding size, default: 300')
    parser.add_argument('--num_layers', default=2, type=int, metavar='N', help='number of layers in rnn decoder. Default: 4')
    parser.add_argument('--dp', default=0.25, type=float, metavar='N', help='dropout probability, default: 0.25')
    parser.add_argument('--bi', type=str2bool, default=True,
                        help='Use bidrectional encoder, default: True')
    parser.add_argument('--reverse_input', type=str2bool, default=False,
                        help='Reverse the input to the encoder. Default: False')
    parser.add_argument('--v', default=30000, type=int, metavar='N',
                        help='Vocabulary size. Use 0 for max size. Default: 30000')
    parser.add_argument('--b', default=64, type=int, metavar='N', help='Batch size, default: 64')
    parser.add_argument('--epochs', default=80, type=int, metavar='N', help='number of epochs, default: 80')
    parser.add_argument('--max_len', type=int, metavar="N", default=30, help="Truncate the sequences to the given max_len parameter.")
    parser.add_argument('--corpus', nargs='+',  default="europarl", metavar='STR',
                        help="Please pass one or more valid corpora, e.g. europarl ted2013 tatoeba")
    parser.add_argument('--attn', default="dot", type=str, help="Attention type: dot, none. Default: dot")
    parser.add_argument('--lang_code', metavar='STR', default="de",
                        help="Provide language code, e.g. 'de'. This is the second language. First is by default English. Default: 'de'")
    parser.add_argument('--reverse', type=str2bool, default=True,
                        help="Reverse language combination. By default the direction is set to EN > lang_code. Set True if you want to train lang_code > EN. Default: True.")
    parser.add_argument('--cuda', type=str2bool, default="True", help="True if model should be trained on GPU, else False. Default: True")
    parser.add_argument('--rnn', metavar="STR", default="lstm", help="Select the rnn type. Possible values: gru and lstm. Default: lstm.")
    parser.add_argument('--train', default=170000, type=int, help="Number of training examples")
    parser.add_argument('--val', default=1020, type=int, help="Number of validation examples")
    parser.add_argument('--test', default=1190, type=int, help="Number of test examples")
    parser.add_argument('--data_dir', default=None, type=str, help="Data directory. Provide this, if data are not in the default data directory of the project. Default: None.")
    parser.add_argument('--tok', default="tok", type=str, help="Infix of tokenized files (e.g. train.tok.de), or specify other: train.de ('')")
    parser.add_argument('--min', type=int, default=5,
                        help="Minimal word frequency. If min_freq <= 0, then min_freq is set to default value")
    parser.add_argument('--tied', default="False", type=str2bool,
                        help="Tie weights between input and output in decoder.")
    parser.add_argument('--beam', type=int, default=5, help="Beam size used during the model validation.")
    parser.add_argument('--norm', type=float, default=-1.0, help="Check norm during training epochs. Default: False (no check).")
    return parser