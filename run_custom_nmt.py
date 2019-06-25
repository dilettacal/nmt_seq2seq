"""
This script runs a customized experiment.

"""

import os, datetime, time, sys

import torch
import torch.nn as nn

from project.experiment.setup_experiment import experiment_parser, Experiment
from project.model.models import Seq2Seq
from project.utils.constants import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from project.utils.data.vocabulary import get_vocabularies_iterators, print_data_info
from project.utils.training import validate, train_model
from project.utils.utils import convert, Logger
from settings import MODEL_STORE
import math


def main():

    experiment = Experiment()
    print("Running experiment on:", experiment.get_device())
   # args.corpus = "europarl"
    model_type = "custom"
    src_lang = experiment.get_src_lang()
    trg_lang = experiment.get_trg_lang()

    print("Language combination ({}-{})".format(src_lang, trg_lang))

    lang_comb = "{}_{}".format(src_lang, trg_lang)
    layers = experiment.nlayers

    experiment.bi = True if (experiment.bi and not experiment.reverse_input) else False

    direction = "bi" if experiment.bi else "uni"

    experiment_path = os.path.join(MODEL_STORE, lang_comb, model_type, str(layers),
                                   direction,
                                   datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(experiment_path, exist_ok=True)


    # Load and process data
    time_data = time.time()
    SRC, TRG, train_iter, val_iter, test_iter, train_data, val_data, test_data = \
        get_vocabularies_iterators(src_lang, experiment)

    print('Loaded data. |SRC| = {}, |TRG| = {}, Time: {}.'.format(len(SRC.vocab), len(TRG.vocab), convert(time.time() - time_data)))

    experiment.src_vocab_size = len(SRC.vocab)
    experiment.trg_vocab_size = len(TRG.vocab)
    data_logger = Logger(path=experiment_path, file_name="data.log")
    print_data_info(data_logger, train_data, val_data, test_data, SRC, TRG, experiment.corpus)

    # Load embeddings if available
    # Create model

    tokens_bos_eos_pad_unk = [TRG.vocab.stoi[SOS_TOKEN], TRG.vocab.stoi[EOS_TOKEN], TRG.vocab.stoi[PAD_TOKEN], TRG.vocab.stoi[UNK_TOKEN]]
    ### src_vocab_size, trg_vocab_size, emb_size, h_dim, num_layers, dropout_p, bi, rnn_type="lstm", tokens_bos_eos_pad_unk=[0, 1, 2, 3],  device=DEFAULT_DEVICE
    model = Seq2Seq(experiment, tokens_bos_eos_pad_unk)
    print(model)

    # Load pretrained model

    #if args.model is not None and os.path.isfile(args.model):
     #   model.load_state_dict(torch.load(args.model))
      #  print('Loaded pretrained model.')

    model = model.to(experiment.get_device())

    # Create weight to mask padding tokens for loss function
    weight = torch.ones(len(TRG.vocab))
    weight[TRG.vocab.stoi[PAD_TOKEN]] = 0
    weight = weight.to(experiment.get_device())

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=experiment.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.25, verbose=True,
                                                           cooldown=6)

    # Create directory for logs, create logger, log hyperparameters
    logger = Logger(experiment_path)
    logger.log('COMMAND ' + ' '.join(sys.argv), stdout=False)
    logger.log('ARGS: {}\nOPTIMIZER: {}\nLEARNING RATE: {}\nSCHEDULER: {}\nMODEL: {}\n'.format(experiment.get_args(), optimizer, experiment.lr,
                                                                                               vars(scheduler), model),
               stdout=False)

    results_logger = Logger(experiment_path, file_name="results.log")
    start_time = time.time()

    """
    Training the model
    
    """

    #train_iter, val_iter, model, criterion, optimizer, scheduler, epochs, logger=None, device=DEFAULT_DEVICE
    train_model(train_iter, val_iter, model, criterion, optimizer, scheduler,TRG=TRG, epochs=experiment.epochs, logger=logger, device=experiment.get_device())

    """
    Validation on test set
    Performed with different beam sizes:
    - Size 1, Greedy Search
    - Size 2, 5 and 12
    """

    ### Evaluation on test set
    beam_size = 1
    logger.log("Validation of test set - Beam size: {}".format(beam_size))
    val_loss, bleu= validate(val_iter=test_iter, model=model, criterion=criterion, device=experiment.get_device(), TRG=TRG, beam_size=beam_size)
    logger.log(
        f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f} | Val. BLEU: {bleu:.3f}')

    beam_size = 2
    logger.log("Validation of test set - Beam size: {}".format(beam_size))
    val_loss, bleu = validate(val_iter=test_iter, model=model, criterion=criterion, device=experiment.get_device(),
                              TRG=TRG, beam_size=beam_size)
    logger.log(
        f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f} | Val. BLEU: {bleu:.3f}')


    beam_size = 5
    logger.log("Validation of test set - Beam size: {}".format(beam_size))
    val_loss, bleu = validate(val_iter=test_iter, model=model, criterion=criterion, device=experiment.get_device(), TRG=TRG, beam_size=beam_size)
    logger.log(
        f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f} | Val. BLEU: {bleu:.3f}')

    beam_size = 12
    logger.log("Validation of test set - Beam size: {}".format(beam_size))
    val_loss, bleu = validate(val_iter=test_iter, model=model, criterion=criterion, device=experiment.get_device(), TRG=TRG, beam_size=beam_size)
    logger.log(
        f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f} | Val. BLEU: {bleu:.3f}')

    logger.log('Finished in {}'.format(convert(time.time() - start_time)))
    return


if __name__ == '__main__':
    print(' '.join(sys.argv))
    main()
