"""
Main script to run nmt experiments
"""
import os, datetime, time, sys

import torch
import torch.nn as nn
import numpy as np

from project.utils.experiment import Experiment
from project.model.models import count_trainable_params, get_nmt_model
from project.utils.constants import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from project.utils.parsers.get_train_model_parser import experiment_parser
from project.utils.train_preprocessing import get_vocabularies_and_iterators, print_info, count_unks
from project.utils.training import train_model, beam_predict, check_translation, CustomReduceLROnPlateau
from project.utils.utils import convert_time_unit, Logger, Metric
from settings import MODEL_STORE


def main():
    TRAIN_SUB_SAMPLE = 170000
    experiment = Experiment(experiment_parser())
    print("Running experiment on:", experiment.get_device())
    # Model configuration
    if experiment.attn != "none":
        experiment.model_type = "custom"
    else:
        experiment.model_type = "s"

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
    logger.log("Attention: {}".format(experiment.attn))


    # Load and process data
    time_data = time.time()
    SRC, TRG, train_iter, val_iter, test_iter, train_data, val_data, test_data, samples, samples_iter = \
        get_vocabularies_and_iterators(experiment, data_dir)
    end_time_data = time.time()

    # Pickle vocabulary objects
    logger.pickle_obj(SRC, "src")
    logger.pickle_obj(TRG, "trg")
    logger.log("SRC and TRG objects persisted in the experiment directory.")

    experiment.src_vocab_size = len(SRC.vocab)
    experiment.trg_vocab_size = len(TRG.vocab)
    data_logger = Logger(path=experiment_path, file_name="data.log")
    translation_logger = Logger(path=experiment_path, file_name="train_translations.log")

    samples_quantities = [170000, 1020, 1190]
    full_quantities = [0,5546,6471] #0 means whole training dataset

    args_quatities = [experiment.train_samples, experiment.val_samples, experiment.test_samples]
    print(args_quatities)

    if args_quatities != samples_quantities and args_quatities != full_quantities:
        ### creates a data.log only if training is not performed on the whole dataset or on the subsample of 170000 sentences
        print_info(data_logger, train_data, val_data, test_data, val_iter, test_iter, SRC, TRG, experiment)
    # Create model
    # special tokens
    tokens_bos_eos_pad_unk = [TRG.vocab.stoi[SOS_TOKEN], TRG.vocab.stoi[EOS_TOKEN], TRG.vocab.stoi[PAD_TOKEN], TRG.vocab.stoi[UNK_TOKEN]]

    model = get_nmt_model(experiment, tokens_bos_eos_pad_unk)
    print(model)
    model = model.to(experiment.get_device())

    # Criterion
    # Masking loss: https://discuss.pytorch.org/t/how-can-i-compute-seq2seq-loss-using-mask/861/21
    weight = torch.ones(len(TRG.vocab))
    weight[TRG.vocab.stoi[PAD_TOKEN]] = 0
    weight = weight.to(experiment.get_device())

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=weight) # or ignore_index = TRG.vocab.stoi[PAD_TOKEN]
    # Optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=experiment.lr)

    # Scheduler
    SCHEDULER_PATIENCE = 15
   # MIN_LR = 2e-07
    MIN_LR = float(np.float(experiment.lr).__mul__(np.float(0.001)))
    logger.log("Scheduler tolerance: {} epochs. Minimal learing rate: {}".format(SCHEDULER_PATIENCE, MIN_LR))
    scheduler = CustomReduceLROnPlateau(optimizer, 'max', patience=SCHEDULER_PATIENCE, verbose=True, min_lr=MIN_LR, factor=0.1)


    logger.log('|src_vocab| = {}, |trg_vocab| = {}, Data Loading Time: {}.'.format(len(SRC.vocab), len(TRG.vocab),
                                                                                   convert_time_unit(end_time_data - time_data)))
    logger.log(">>>> Path to model: {}".format(os.path.join(logger.path, "model.pkl")))
    logger.log('CLI-ARGS ' + ' '.join(sys.argv), stdout=False)
    logger.log('Args: {}\nOPTIM: {}\nLR: {}\nSCHED: {}\nMODEL: {}\n'.format(experiment.get_args(), optimizer, experiment.lr, vars(scheduler), model), stdout=False)
    logger.log(f'Trainable parameters: {count_trainable_params(model):,}')

    logger.pickle_obj(experiment.get_dict(), "experiment")

    start_time = time.time()

    # Train the model

    log_every = 5
    bleu, metrics = train_model(train_iter=train_iter, val_iter=val_iter, model=model, criterion=criterion,
                                optimizer=optimizer, scheduler=scheduler, epochs=experiment.epochs, SRC=SRC, TRG=TRG,
                                logger=logger, device=experiment.get_device(), tr_logger=translation_logger,
                                samples_iter=samples_iter, check_translations_every=log_every,
                                beam_size=experiment.val_beam_size, clip_value=experiment.get_clip_value())

    # Uncomment following lines if you want to pickle metric results and/or plot bleus and losses
    #nltk_bleu_metric = Metric("nltk_bleu", list(bleu.values())[0])
    #train_loss = Metric("train_loss", list(metrics.values())[0])
    #train_bleus = dict({"train": train_loss.values, "bleu": nltk_bleu_metric.values})
    #logger.plot(train_bleus, title="Train Loss vs. Val BLEU", ylabel="Loss/BLEU", file="loss_bleu")

    FIXED_WORD_LEVEL_LEN = 30

    max_len = FIXED_WORD_LEVEL_LEN

    # Test the model on the test dataset

    # Beam 1
    logger.log("Validation of test set")
    beam_size = 1
    logger.log("Prediction of test set - Beam size: {}".format(beam_size))
    bleu = beam_predict(model, val_iter, experiment.get_device(), beam_size, TRG, max_len=max_len)
    logger.log(f'\t Test. (nltk) BLEU: {bleu:.3f}')

    # Beam 5
    beam_size = 5
    logger.log("Prediction of test set - Beam size: {}".format(beam_size))
    bleu = beam_predict(model, val_iter, experiment.get_device(), beam_size, TRG, max_len=max_len)
    logger.log(f'\t Test. (nltk) BLEU: {bleu:.3f}')

    # Beam 10
    beam_size = 10
    logger.log("Prediction of test set - Beam size: {}".format(beam_size))
    bleu = beam_predict(model, val_iter, experiment.get_device(), beam_size, TRG, max_len=max_len)
    logger.log(f'\t Test. (nltk) BLEU: {bleu:.3f}')

    # Translate some sentences
    final_translation = Logger(file_name="final_translations.log", path=experiment_path)
    check_translation(samples=samples_iter, model=model, SRC=SRC, TRG=TRG, logger=final_translation, persist=True)

    logger.log('Finished in {}'.format(convert_time_unit(time.time() - start_time)))

    return

if __name__ == '__main__':
    print(' '.join(sys.argv))
    main()

