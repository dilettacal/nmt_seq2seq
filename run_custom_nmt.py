"""
Main script to run nmt experiments

"""
import argparse
import os, datetime, time, sys

import torch
import torch.nn as nn

from project.utils.experiment import Experiment
from project.model.models import count_parameters, get_nmt_model
from project.utils.constants import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from project.utils.vocabulary import get_vocabularies_iterators, print_info
from project.utils.training import train_model, beam_predict, check_translation
from project.utils.utils import convert_time_unit, Logger, Metric, str2bool
from settings import MODEL_STORE



def experiment_parser():
    parser = argparse.ArgumentParser(description='Neural Machine Translation')
    parser.add_argument('--lr', default=2e-3, type=float, metavar='N', help='learning rate, default: 2e-3')
    parser.add_argument('--hs', default=300, type=int, metavar='N', help='size of hidden state, default: 300')
    parser.add_argument('--emb', default=300, type=int, metavar='N', help='embedding size, default: 300')
    parser.add_argument('--nlayers', default=2, type=int, metavar='N', help='number of layers in rnn, default: 2')
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

    parser.add_argument('--train', default=170000, type=int, help="Number of training examples")
    parser.add_argument('--val', default=1020, type=int, help="Number of validation examples")
    parser.add_argument('--test', default=1190, type=int, help="Number of test examples")
    parser.add_argument('--data_dir', default=None, type=str, help="Data directory")
    parser.add_argument('--tok', default="tok", type=str, help="File name: train.tok.de, or specify other: train.de ('')")
    parser.add_argument('--min', type=int, default=5,
                        help="Minimal word frequency. If min_freq < 0, then min_freq is set to default value")
    parser.add_argument('--tied', default="False", type=str2bool,
                        help="Tie weights between input and output in decoder.")
    parser.add_argument('--pretrained', default="False", type=str2bool,
                        help="Initialize embeddings weights with pre-trained embeddings")

    parser.add_argument('--beam', type=int, default=5, help="Beam size used during the model validation.")
    return parser

def main():

    experiment = Experiment(experiment_parser())
    print("Running experiment on:", experiment.get_device())
    ### define model type:

    if experiment.pretrained or experiment.attn != "none":
        experiment.model_type = "custom"
        experiment.reverse_input = False
        experiment.bi = True
    else:
        experiment.model_type = "s"
        if experiment.bi and experiment.reverse_input:
            experiment.reverse_input = False

    model_type = experiment.model_type
    print("Model Type", model_type)
    src_lang = experiment.get_src_lang()
    trg_lang = experiment.get_trg_lang()


    lang_comb = "{}_{}".format(src_lang, trg_lang)
    layers = experiment.nlayers

    direction = "bi" if experiment.bi else "uni"

    rnn_type = experiment.rnn_type
    experiment_path = os.path.join(MODEL_STORE, lang_comb, model_type, rnn_type, str(layers),
                                   direction,
                                   datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(experiment_path, exist_ok=True)

    data_dir = experiment.data_dir

    # Create directory for logs, create logger, log hyperparameters
    logger = Logger(experiment_path)
    logger.log("Language combination ({}-{})".format(src_lang, trg_lang))


    # Load and process data
    time_data = time.time()
    SRC, TRG, train_iter, val_iter, test_iter, train_data, val_data, test_data, samples, samples_iter = \
        get_vocabularies_iterators(experiment, data_dir)

    end_time_data = time.time()

    logger.pickle_obj(SRC, "src")
    logger.pickle_obj(TRG, "trg")
    logger.log("SRC and TRG objects persisted in the experiment directory.")

    experiment.src_vocab_size = len(SRC.vocab)
    experiment.trg_vocab_size = len(TRG.vocab)
    data_logger = Logger(path=experiment_path, file_name="data.log")
    translation_logger = Logger(path=experiment_path, file_name="train_translations.log")

    print_info(data_logger, train_data, val_data, test_data, SRC, TRG, experiment)

    # Create model
    tokens_bos_eos_pad_unk = [TRG.vocab.stoi[SOS_TOKEN], TRG.vocab.stoi[EOS_TOKEN], TRG.vocab.stoi[PAD_TOKEN], TRG.vocab.stoi[UNK_TOKEN]]

    if experiment.pretrained:
        pretraiend_src = train_data.fields['src'].vocab.vectors
        pretrained_trg = train_data.fields['trg'].vocab.vectors
        logger.log("Running experiment with pretrained embeddings!")
    else:
        pretraiend_src, pretrained_trg = None, None

    model = get_nmt_model(experiment, tokens_bos_eos_pad_unk)
    print(model)
    if experiment.pretrained:
        model.load_pretrained_embeddings(pretraiend_src, pretrained_trg)
        logger.log("Pretrained embeddings loaded!")

    model = model.to(experiment.get_device())

    # Create weight to mask padding tokens for loss function
    weight = torch.ones(len(TRG.vocab))
    weight[TRG.vocab.stoi[PAD_TOKEN]] = 0
    weight = weight.to(experiment.get_device())

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=experiment.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20, verbose=True, min_lr=2e-7, factor=0.1)


    logger.log('|src_vocab| = {}, |trg_vocab| = {}, Data Loading Time: {}.'.format(len(SRC.vocab), len(TRG.vocab),
                                                                                   convert_time_unit(end_time_data - time_data)))
    logger.log(">>>> Path to model: {}".format(os.path.join(logger.path, "model.pkl")))
    logger.log('CLI-ARGS ' + ' '.join(sys.argv), stdout=False)
    logger.log('Args: {}\nOPTIM: {}\nLR: {}\nSCHED: {}\nMODEL: {}\n'.format(experiment.get_args(), optimizer, experiment.lr,
                                                                                               vars(scheduler), model),
               stdout=False)

    logger.log(f'Trainable parameters: {count_parameters(model):,}')

    logger.pickle_obj(experiment.get_dict(), "experiment")

    start_time = time.time()

    """
    Training the model
    
    """
    log_every=5
    bleus, metrics = train_model(train_iter=train_iter, val_iter=val_iter, model=model, criterion=criterion,
                                 optimizer=optimizer, scheduler=scheduler, epochs=experiment.epochs, SRC=SRC, TRG=TRG,
                                 logger=logger, device=experiment.get_device(), tr_logger=translation_logger,
                                 samples_iter=samples_iter, check_translations_every=log_every,
                                 beam_size=experiment.val_beam_size, char_level=experiment.char_level)

    nltk_bleu_metric = Metric("nltk_bleu", list(bleus.values())[0])
    train_loss = Metric("train_loss", list(metrics.values())[0])
    train_perpl = Metric("train_ppl", list(metrics.values())[1])

    logger.pickle_obj(nltk_bleu_metric.get_dict(), "nltk_bleus")
    logger.pickle_obj(train_loss.get_dict(), "train_losses")
    logger.pickle_obj(train_perpl.get_dict(), "train_ppl")

    ### plot ppl and bleu + train and bleu
    ppl_bleus = dict({"ppl": train_perpl.values, "bleu": nltk_bleu_metric.values})
    train_bleus = dict({"train": train_loss.values, "bleu": nltk_bleu_metric.values})

    logger.plot(ppl_bleus, title="Train PPL vs. Val BLEU", ylabel="PPL/BLEU", file="ppl_bleu")
    logger.plot(train_bleus, title="Train Loss vs. Val BLEU", ylabel="Loss/BLEU", file="loss_bleu")

    FIXED_WORD_LEVEL_LEN = 30
    FIXED_CHAR_LEVEL_LEN = 1500

    max_len = FIXED_WORD_LEVEL_LEN if experiment.char_level == False else FIXED_CHAR_LEVEL_LEN
    ### Evaluation on test set
    logger.log("Validation of test set")
    beam_size = 1
    logger.log("Prediction of test set - Beam size: {}".format(beam_size))
    bleus = beam_predict(model, val_iter, experiment.get_device(), beam_size, TRG, max_len=max_len, char_level=experiment.char_level)
    nltk_b, perl_b = bleus
    if perl_b == None: perl_b = 0
    logger.log(
        f'\t Test. (nltk) BLEU: {nltk_b:.3f} | Test. (perl) BLEU: {perl_b:.3f}')

    beam_size = 2
    logger.log("Prediction of test set - Beam size: {}".format(beam_size))
    bleus = beam_predict(model, val_iter, experiment.get_device(), beam_size, TRG, max_len=max_len,  char_level=experiment.char_level)
    nltk_b, perl_b = bleus
    if perl_b == None: perl_b = 0
    logger.log(
        f'\t Test. (nltk) BLEU: {nltk_b:.3f} | Test. (perl) BLEU: {perl_b:.3f}')

    beam_size = 5
    logger.log("Prediction of test set - Beam size: {}".format(beam_size))
    bleus = beam_predict(model, val_iter, experiment.get_device(), beam_size, TRG, max_len=max_len,  char_level=experiment.char_level)
    nltk_b, perl_b = bleus
    if perl_b == None: perl_b = 0
    logger.log(
        f'\t Test. (nltk) BLEU: {nltk_b:.3f} | Test. (perl) BLEU: {perl_b:.3f}')

    beam_size = 10
    logger.log("Prediction of test set - Beam size: {}".format(beam_size))
    bleus = beam_predict(model, val_iter, experiment.get_device(), beam_size, TRG, max_len=max_len,  char_level=experiment.char_level)
    nltk_b, perl_b = bleus
    if perl_b == None: perl_b = 0
    logger.log(
        f'\t Test. (nltk) BLEU: {nltk_b:.3f} | Test. (perl) BLEU: {perl_b:.3f}')

    logger.log('Finished in {}'.format(convert_time_unit(time.time() - start_time)))


    ########## testing model against all samples translations

    final_translation = Logger(file_name="final_translations.log", path=experiment_path)
    check_translation(model=model, SRC=SRC, TRG=TRG, logger=final_translation, samples=samples_iter, persist=True, char_level=experiment.char_level)

    return

if __name__ == '__main__':
    print(' '.join(sys.argv))
    main()

