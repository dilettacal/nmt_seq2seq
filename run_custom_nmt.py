"""
Main script to run nmt experiments

"""

import os, datetime, time, sys

import torch
import torch.nn as nn

from project.utils.arg_parse import experiment_parser
from project.utils.experiment import Experiment
from project.model.models import count_parameters, get_nmt_model
from project.utils.constants import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from project.utils.vocabulary import get_vocabularies_iterators, print_data_info
from project.utils.training import train_model, validate_test_set, beam_predict
from project.utils.utils import convert, Logger, Metric, load_embeddings
from settings import MODEL_STORE

def main():

    experiment = Experiment(experiment_parser())

    MAX_LEN = experiment.truncate
    print("Running experiment on:", experiment.get_device())
   # args.corpus = "europarl"
    model_type = experiment.model_type
    print("Model Type", model_type)
    src_lang = experiment.get_src_lang()
    trg_lang = experiment.get_trg_lang()


    lang_comb = "{}_{}".format(src_lang, trg_lang)
    layers = experiment.nlayers

    experiment.bi = True if (experiment.bi and not experiment.reverse_input) else False

    direction = "bi" if experiment.bi else "uni"

    experiment_path = os.path.join(MODEL_STORE, lang_comb, model_type, str(layers),
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

    experiment.src_vocab_size = len(SRC.vocab)
    experiment.trg_vocab_size = len(TRG.vocab)
    data_logger = Logger(path=experiment_path, file_name="data.log")
    translation_logger = Logger(path=experiment_path, file_name="translations.log")

    print_data_info(data_logger, train_data, val_data, test_data, SRC, TRG, experiment)

    # Create model
    tokens_bos_eos_pad_unk = [TRG.vocab.stoi[SOS_TOKEN], TRG.vocab.stoi[EOS_TOKEN], TRG.vocab.stoi[PAD_TOKEN], TRG.vocab.stoi[UNK_TOKEN]]

    if experiment.pretrained:
        np_lang_code_file = 'scripts/emb-{}-{}.npy'.format(len(SRC.vocab), experiment.lang_code)
        np_en_file = 'scripts/emb-{}-en.npy'.format(len(TRG.vocab))
        embedding_lang_code, embedding_EN = load_embeddings(np_lang_code_file, np_en_file)
        if experiment.src_lang == "en":
            pretraiend_src = embedding_EN
            pretrained_trg = embedding_lang_code
        else:
            pretraiend_src = embedding_lang_code
            pretrained_trg = embedding_EN
        logger.log("Running experiment with pretrained embeddings!")
    else:
        pretraiend_src, pretrained_trg = None, None

    model = get_nmt_model(experiment, tokens_bos_eos_pad_unk, pretraiend_src, pretrained_trg)
    print(model)
    model = model.to(experiment.get_device())

    # Create weight to mask padding tokens for loss function
    weight = torch.ones(len(TRG.vocab))
    weight[TRG.vocab.stoi[PAD_TOKEN]] = 0
    weight = weight.to(experiment.get_device())

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=experiment.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20, verbose=True, min_lr=1e-6, cooldown=1, factor=0.25)


    logger.log('Loaded data. |SRC| = {}, |TRG| = {}, Time: {}.'.format(len(SRC.vocab), len(TRG.vocab),
                                                                       convert(end_time_data - time_data)))
    logger.log(">>>> Path to model: {}".format(os.path.join(logger.path, "model.pkl")))
    logger.log('COMMAND ' + ' '.join(sys.argv), stdout=False)
    logger.log('ARGS: {}\nOPTIMIZER: {}\nLEARNING RATE: {}\nSCHEDULER: {}\nMODEL: {}\n'.format(experiment.get_args(), optimizer, experiment.lr,
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
                                 optimizer=optimizer, scheduler=scheduler, SRC=SRC, TRG=TRG,
                                 epochs=experiment.epochs, logger=logger, device=experiment.get_device(),
                                 tr_logger=translation_logger, samples_iter=samples_iter, check_translations_every=log_every)

    ### metrics metrics.({"loss": train_losses, "ppl": train_ppls})
    nltk_bleu_metric = Metric("nltk_bleu", list(bleus.values())[0])
  # perl_bleu_metric = Metric("bleu_perl", list(bleus.values())[1])
    train_loss = Metric("train_loss", list(metrics.values())[0])
    train_perpl = Metric("train_ppl", list(metrics.values())[1])
    #metric, title, ylabel, file
    #title="Validation nltk BLEU/Epochs", ylabel="BLEU", file="nltk_bleu"
   # logger.plot(dict({"Train PPL": list(metrics.values())[1], "BLEU": list(bleus.values())[0]}),
    #            title="Training PPL/BLEU over the epochs", ylabel="PPL/BLEU",
     #           file="train_metrics", log_every=log_every)


    logger.pickle_obj(nltk_bleu_metric.get_dict(), "nltk_bleus")
   # logger.pickle_obj(perl_bleu_metric.get_dict(), "perl_bleus")
    logger.pickle_obj(train_loss.get_dict(), "train_losses")
    logger.pickle_obj(train_perpl.get_dict(), "train_ppl")



    ### Evaluation on test set
    logger.log("Validation of test set")
    beam_size = 1
    logger.log("Prediction of test set - Beam size: {}".format(beam_size))
    bleus = beam_predict(model, val_iter, experiment.get_device(), beam_size, TRG, max_len=30)
    nltk_b, perl_b = bleus
    logger.log(
        f'\t Test. (nltk) BLEU: {nltk_b:.3f} | Test. (perl) BLEU: {perl_b:.3f}')

    beam_size = 2
    logger.log("Prediction of test set - Beam size: {}".format(beam_size))
    bleus = beam_predict(model, val_iter, experiment.get_device(), beam_size, TRG, max_len=30)
    nltk_b, perl_b = bleus
    logger.log(
        f'\t Test. (nltk) BLEU: {nltk_b:.3f} | Test. (perl) BLEU: {perl_b:.3f}')

    beam_size = 5
    logger.log("Prediction of test set - Beam size: {}".format(beam_size))
    bleus = beam_predict(model, val_iter, experiment.get_device(), beam_size, TRG, max_len=30)
    nltk_b, perl_b = bleus
    logger.log(
        f'\t Test. (nltk) BLEU: {nltk_b:.3f} | Test. (perl) BLEU: {perl_b:.3f}')

    beam_size = 10
    logger.log("Prediction of test set - Beam size: {}".format(beam_size))
    bleus = beam_predict(model, val_iter, experiment.get_device(), beam_size, TRG, max_len=30)
    nltk_b, perl_b = bleus
    logger.log(
        f'\t Test. (nltk) BLEU: {nltk_b:.3f} | Test. (perl) BLEU: {perl_b:.3f}')

    logger.log('Finished in {}'.format(convert(time.time() - start_time)))


    return

if __name__ == '__main__':
    print(' '.join(sys.argv))
    main()
