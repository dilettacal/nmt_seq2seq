import argparse


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


def experiment_parser():
    parser = argparse.ArgumentParser(description='Neural Machine Translation')
    parser.add_argument('--lr', default=2e-3, type=float, metavar='N', help='learning rate, default: 2e-3')
    parser.add_argument('--hs', default=300, type=int, metavar='N', help='size of hidden state, default: 300')
    parser.add_argument('--emb', default=300, type=int, metavar='N', help='embedding size, default: 300')
    parser.add_argument('--nlayers', default=2, type=int, metavar='N', help='number of layers in rnn, default: 2')
    parser.add_argument('--dp', default=0.30, type=float, metavar='N', help='dropout probability, default: 0.30')
    parser.add_argument('--bi', type=str2bool, default=False,
                        help='use bidrectional encoder, default: false')
    # parser.add_argument('--attn', default=None, type=str, metavar='STR',
    #  help='attention: dot-product, additive or none, default: dot-product ')
    parser.add_argument('--reverse_input', dest='reverse_input', type=str2bool, default=False,
                        help='reverse input to encoder, default: False')
    parser.add_argument('-v', default=0, type=int, metavar='N', help='vocab size, use 0 for maximum size, default: 0')
    parser.add_argument('-b', default=64, type=int, metavar='N', help='batch size, default: 64')
    parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of epochs, default: 50')
    parser.add_argument('--model', metavar='DIR', default=None, help='path to model, default: None')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='only evaluate model, default: False')
    # parser.add_argument('--visualize', dest='visualize', action='store_true', help='visualize model attention distribution')
    parser.add_argument('--predict', metavar='DIR', default=None,
                        help='directory with final input data for predictions, default: None')
    parser.add_argument('--predict_outfile', metavar='DIR', default='data/preds.txt',
                        help='file to output final predictions, default: "data/preds.txt"')
    parser.add_argument('--predict_from_input', metavar='STR', default=None, help='Source sentence to translate')
    parser.add_argument('--max_len', type=int, metavar="N", default=30, help="Sequence max length. Default 30 units.")
    parser.add_argument('--model_type', default="custom", metavar='STR', help="Model type (custom, cho, sutskever)")

    parser.add_argument('--corpus', default="", metavar='STR',
                        help="The corpus, where training should be performed. Possible values: \'europarl\' and \'simple'\ - the iwslt dataset from torchtext")

    parser.add_argument('-c', metavar='STR', default=False, help="Training at char level")

    parser.add_argument('-lang_code', metavar='STR', default="de",
                        help="Provide language code, e.g. 'de'. This is the source or target language.")
    parser.add_argument('--reverse', metavar="STR", default=False,
                        help="Reverse language combination. Standard: EN > <lang>, if reverse, then <lang> > EN")

    parser.add_argument('--cuda', metavar='STR', default=True)

    parser.add_argument('--tie', metavar='STR', default=False)

    parser.add_argument('--rnn', metavar="STR", default="lstm")

    # parser.set_defaults(evaluate=False, bi=False, reverse_input=False, visualize=False)
    parser.set_defaults(evaluate=False)