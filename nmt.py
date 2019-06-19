import argparse, os, datetime, time, sys

import torch
import torch.nn as nn

from project.model.models import Seq2Seq, ChoSeq2Seq, VALID_MODELS
from project.utils.constants import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from project.utils.data.vocabulary import prepare_dataset_iterators, print_data_info
from project.utils.training import predict_from_input, predict, validate, train
from project.utils.utils import convert, Logger
from settings import MODEL_STORE

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



parser = argparse.ArgumentParser(description='Neural Machine Translation')
parser.add_argument('--lr', default=2e-3, type=float, metavar='N', help='learning rate, default: 2e-3')
parser.add_argument('--hs', default=300, type=int, metavar='N', help='size of hidden state, default: 300')
parser.add_argument('--emb', default=300, type=int, metavar='N', help='embedding size, default: 300')
parser.add_argument('--nlayers', default=2, type=int, metavar='N', help='number of layers in rnn, default: 2')
parser.add_argument('--dp', default=0.30, type=float, metavar='N', help='dropout probability, default: 0.30')
parser.add_argument('--bi',type=str2bool, default=False,
                    help='use bidrectional encoder, default: false')
#parser.add_argument('--attn', default=None, type=str, metavar='STR',
                  #  help='attention: dot-product, additive or none, default: dot-product ')
parser.add_argument('--reverse_input', dest='reverse_input', type=str2bool, default=False,
                    help='reverse input to encoder, default: False')
parser.add_argument('-v', default=0, type=int, metavar='N', help='vocab size, use 0 for maximum size, default: 0')
parser.add_argument('-b', default=64, type=int, metavar='N', help='batch size, default: 64')
parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of epochs, default: 50')
parser.add_argument('--model', metavar='DIR', default=None, help='path to model, default: None')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='only evaluate model, default: False')
#parser.add_argument('--visualize', dest='visualize', action='store_true', help='visualize model attention distribution')
parser.add_argument('--predict', metavar='DIR', default=None,
                    help='directory with final input data for predictions, default: None')
parser.add_argument('--predict_outfile', metavar='DIR', default='data/preds.txt',
                    help='file to output final predictions, default: "data/preds.txt"')
parser.add_argument('--predict_from_input', metavar='STR', default=None, help='Source sentence to translate')
parser.add_argument('--max_len', type=int, metavar="N", default=30, help="Sequence max length. Default 30 units.")
parser.add_argument('--model_type', default="custom",metavar='STR',help="Model type (custom, cho, sutskever)")

parser.add_argument('--corpus', default="",metavar='STR', help="The corpus, where training should be performed. Possible values: \'europarl\' and \'simple'\ - the iwslt dataset from torchtext")

parser.add_argument('-c', metavar='STR',  default=False, help="Training at char level")

parser.add_argument('-lang_code', metavar='STR', default="de", help="Provide language code, e.g. 'de'. This is the source or target language.")
parser.add_argument('--reverse', metavar="STR", default=False, help="Reverse language combination. Standard: EN > <lang>, if reverse, then <lang> > EN")

parser.add_argument('--cuda', metavar='STR', default=True)

parser.add_argument('--tie', metavar='STR', default = False)

parser.add_argument('--rnn', metavar="STR", default="lstm")

#parser.set_defaults(evaluate=False, bi=False, reverse_input=False, visualize=False)
parser.set_defaults(evaluate=False)


def main():
    global args
    args = parser.parse_args()
    corpus = args.corpus
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
   # device = "cpu"

    print("Running experiment on:", device)

    # models are stored in <root>/results
    # results/de_en, results/en_de

    voc_limit = args.v
    lang_code = args.lang_code
    tie_emb = args.tie
    rnn_type = args.rnn
    src_lang = lang_code if args.reverse else "en"
    trg_lang = "en" if src_lang == lang_code else lang_code
    char_level = args.c
    model_type = args.model_type.lower()

    print("Language combination ({}-{})".format(src_lang, trg_lang))

    lang_comb = "{}_{}".format(src_lang, trg_lang)
    layers = args.nlayers
    direction = "bi" if args.bi else "uni"

    experiment_path = os.path.join(MODEL_STORE, lang_comb, model_type, str(layers), direction, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(experiment_path, exist_ok=True)


    # Load and process data
    time_data = time.time()
    SRC, TRG, train_iter, val_iter, test_iter, train_data, val_data, test_data = \
        prepare_dataset_iterators(v=voc_limit, corpus=corpus,
                                  language_code=lang_code,
                                  batch_size=args.b,
                                  max_sent_len=args.max_len,
                                  src_l=src_lang,
                                  device=device, char_level=char_level)

    print('Loaded data. |SRC| = {}, |TRG| = {}, Time: {}.'.format(len(SRC.vocab), len(TRG.vocab), convert(time.time() - time_data)))
    print("First train src batch:", SRC.reverse(next(iter(train_iter)).src), sep="\n")
    print("First train trg batch:", TRG.reverse(next(iter(train_iter)).trg), sep="\n")

    data_logger = Logger(path=experiment_path, file_name="data.log")
    print_data_info(data_logger, train_data, val_data, test_data, SRC, TRG, corpus)

    # Load embeddings if available
    LOAD_EMBEDDINGS = False
    if LOAD_EMBEDDINGS:
        # TODO: Change this if pretrained embeddings are used
        embedding_src = None
        embedding_trg = None
    else:
        embedding_src = (torch.rand(len(SRC.vocab), args.emb) - 0.5) * 2
        embedding_trg = (torch.rand(len(TRG.vocab), args.emb) - 0.5) * 2
        print('Initialized embedding vectors')

    # Create model
    tokens = [TRG.vocab.stoi[x] for x in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN,UNK_TOKEN]]
    if model_type not in VALID_MODELS:
        print("Wrong model type. Model type is set to custom")
        model_type = "custom"
    if args.bi and args.reverse_input:
        print("You set both bidirectional and reverse input. Reverse input argument is removed.")
        args.reverse_input = False
    if args.reverse_input:
        model_type = "sutskever"

    ### generate the model ####
    if model_type == "custom":
        model = Seq2Seq(embedding_src=embedding_src,
                        embedding_trg=embedding_trg,
                        h_dim=args.hs,
                        num_layers=layers,
                        dropout_p=args.dp,
                        bi=args.bi,
                        rnn_type=rnn_type,
                        tie_emb=tie_emb,
                        tokens_bos_eos_pad_unk=tokens,
                        reverse_input=args.reverse_input,
                        device=device)
    elif model_type == "sutskever":
        if not args.reverse_input:
            print("Input must be reversed.")
            args.reverse_input = True
        if args.nlayers < 2:
            print("Architecture is set to be multilayered!")
            args.layers = 2
        if rnn_type != "lstm":
            print("Cell type is set to lstm")
            rnn_type = "lstm"
        if args.bi:
            print("Cell is set to be uni directional")
            args.bi = False

        model = Seq2Seq(embedding_src=embedding_src,
                        embedding_trg=embedding_trg,
                        h_dim=args.hs,
                        num_layers=layers,
                        dropout_p=args.dp,
                        bi=args.bi,
                        rnn_type=rnn_type,
                        tie_emb=tie_emb,
                        tokens_bos_eos_pad_unk=tokens,
                        reverse_input=args.reverse_input,
                        device=device)
    elif model_type == "cho":
        model = ChoSeq2Seq(embedding_src=embedding_src,
                        embedding_trg=embedding_trg,
                        h_dim=args.hs,
                        num_layers=layers,
                        dropout_p=args.dp,
                        bi=args.bi,
                        tie_emb=tie_emb,
                        tokens_bos_eos_pad_unk=tokens,
                        reverse_input=args.reverse_input,
                        device=device, maxout_units=100)

    # Load pretrained model
    if args.model is not None and os.path.isfile(args.model):
        model.load_state_dict(torch.load(args.model))
        print('Loaded pretrained model.')

    model = model.to(device)

    # Create weight to mask padding tokens for loss function
    weight = torch.ones(len(TRG.vocab))
    weight[TRG.vocab.stoi[PAD_TOKEN]] = 0
    weight = weight.to(device)

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=30, factor=0.25, verbose=True,
                                                           cooldown=6)

    # Create directory for logs, create logger, log hyperparameters
    logger = Logger(experiment_path)
    logger.log('COMMAND ' + ' '.join(sys.argv), stdout=False)
    logger.log('ARGS: {}\nOPTIMIZER: {}\nLEARNING RATE: {}\nSCHEDULER: {}\nMODEL: {}\n'.format(args, optimizer, args.lr,
                                                                                               vars(scheduler), model),
               stdout=False)
    # Train, validate, or predict
    start_time = time.time()
    if args.predict_from_input is not None:
        predict_from_input(model, args.predict_from_input, SRC, TRG, logger)
    elif args.predict is not None:
        predict(model, args.predict, args.predict_outfile, SRC, TRG, logger)
    #elif args.visualize:
    #    visualize(train_iter, model, SRC, TRG, logger)
    elif args.evaluate:
        validate(val_iter, model, criterion, SRC, TRG, logger, device)
    else:
        train(train_iter, val_iter, model, criterion, optimizer, scheduler, SRC, TRG, args.epochs, logger, device)
    logger.log('Finished in {}'.format(convert(time.time() - start_time)))
    return


if __name__ == '__main__':
    print(' '.join(sys.argv))
    main()
