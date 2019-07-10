import argparse


def str2bool(v):
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
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
    parser.add_argument('--hs', default=512, type=int, metavar='N', help='size of hidden state, default: 300')
    parser.add_argument('--emb', default=256, type=int, metavar='N', help='embedding size, default: 300')
    parser.add_argument('--nlayers', default=4, type=int, metavar='N', help='number of layers in rnn, default: 2')
    parser.add_argument('--dp', default=0.25, type=float, metavar='N', help='dropout probability, default: 0.30')
    parser.add_argument('--bi', type=str2bool, default=False,
                        help='use bidrectional encoder, default: false')
    parser.add_argument('--reverse_input', type=str2bool, default=False,
                        help='reverse input to encoder, default: False')
    parser.add_argument('-v', default=30000, type=int, metavar='N',
                        help='vocab size, use 0 for maximum size, default: 0')
    parser.add_argument('-b', default=64, type=int, metavar='N', help='batch size, default: 64')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of epochs, default: 50')
    parser.add_argument('--model', metavar='DIR', default=None, help='path to model, default: None')

    parser.add_argument('--max_len', type=int, metavar="N", default=30, help="Sequence max length. Default 30 units.")
    parser.add_argument('--model_type', default="custom", metavar='STR', help="Model type (custom, cho, sutskever)")

    parser.add_argument('--corpus', default="europarl", metavar='STR',
                        help="The corpus, where training should be performed. Possible values: \'europarl\' and \'simple'\ - the iwslt dataset from torchtext")

    parser.add_argument('--attn', default="none", type=str, help="Attention type: dot, additive, none")

    parser.add_argument('-c', metavar='STR', default=False, help="Training at char level")

    parser.add_argument('-lang_code', metavar='STR', default="de",
                        help="Provide language code, e.g. 'de'. This is the source or target language.")

    parser.add_argument('--reverse', type=str2bool, default=False,
                        help="Reverse language combination. Standard: EN > <lang>, if reverse, then <lang> > EN")

    parser.add_argument('--cuda', type=str2bool, default="True")

    parser.add_argument('--rnn', metavar="STR", default="lstm")

    parser.add_argument('--train', default=200000, type=int, help="Number of training examples")
    parser.add_argument('--val', default=20000, type=int, help="Number of validation examples")
    parser.add_argument('--test', default=10000, type=int, help="Number of test examples")
    parser.add_argument('--data_dir', default=None, type=str, help="Data directory")
    parser.add_argument('--tok', default="clean", type=str, help="tok files or clean files")
    parser.add_argument('--min', type=int, default=5,
                        help="Minimal word frequency. If min_freq < 0, then min_freq is set to default value")
    parser.add_argument('--tied', default="False", type=str2bool,
                        help="Tie weights between input and output in decoder.")
    parser.add_argument('--pretrained', default="False", type=str2bool,
                        help="Initialize embeddings weights with pre-trained embeddings")
    return parser


def data_prepro_parser():
    parser = argparse.ArgumentParser(description='Neural Machine Translation')
    parser.add_argument("--lang_code", default="de", type=str)
    parser.add_argument("--type", default="tmx", type=str, help="TMX or TXT")
    parser.add_argument("--corpus", default="europarl", type=str, help="Corpus name")
    parser.add_argument("--max_len", default=30, type=int, help="Filter sequences with a length <= max_len")
    parser.add_argument("--min_len", default=1, type=int, help="Filter sequences with a length >= min_len")
    parser.add_argument('--path', default="data/raw/europarl/de", help="Path to raw data files")
    parser.add_argument('--file', default="de-en.tmx", help="File name after extraction")
    parser.add_argument('--v', type=str2bool, default="False", help="Either vocabulary should be reduced by replacing some repeating tokens with labels.\nNumbers are replaced with NUM, Persons names are replaced with PERSON. Require: Spacy!")
    parser.add_argument('--fast', type=str2bool, default="False", help="Filter by fast tokenizing") #False: search for spacy models for the given langugae

    return parser