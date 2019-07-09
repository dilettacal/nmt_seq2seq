import argparse
import datetime
import os
import time

from project import get_full_path
from project.experiment.setup_experiment import Experiment, str2bool
from project.utils.utils import Logger
from project.utils.vocabulary import get_vocabularies_iterators
from settings import MODEL_STORE, DATA_DIR_PREPRO

TRAIN = 170000
VAL = 1020
TEST = 1190

VOC_LIMIT = 30000
MIN_FREQ = 0

root = get_full_path(DATA_DIR_PREPRO)
data_dir = os.path.join(root, "europarl", "de", "splits", str(30))


def experiment_1x(data_logger):
    experiment = Experiment(stats_parser())
    experiment.corpus = "europarl"
    src_lang = experiment.src_lang = "de"
    experiment.trg_lang = "en"
    experiment.lang_code = "de"
    experiment.set_train(170000)
    experiment.set_val(1020)
    experiment.set_test(1190)
    experiment.voc_limit = VOC_LIMIT
    experiment.min_freq = MIN_FREQ

    print(experiment.__dict__)

    SRC, TRG, _, _, _, _, _, _, _, _ = \
        get_vocabularies_iterators(src_lang, experiment, data_dir=data_dir)

    data_logger.log("Data information {} | DE > EN".format(experiment.train_samples))
    data_logger.log(
        "Total samples: {}".format(experiment.train_samples + experiment.val_samples + experiment.test_samples))
    data_logger.log("Total train: {} \t Total val: {} \t Total test: {}")
    data_logger.log("Min frequency: {}".format(experiment.min_freq))
    data_logger.log("Vocabulary limit: {}".format(experiment.voc_limit))

    data_logger.log("Total German words in the vocabulary: {}".format(len(SRC.vocab)))
    data_logger.log("Total English words in the vocabulary: {}".format(len(TRG.vocab)))

    data_logger.log("Total German words in the training dataset: {}".format(sum(SRC.vocab.freqs.values())))
    data_logger.log("Total English words in the training dataset: {}".format(sum(TRG.vocab.freqs.values())))

    #### Min freq 5
    experiment.min_freq = 5
    SRC, TRG, _, _, _, _, _, _, _, _ = \
        get_vocabularies_iterators(src_lang, experiment, data_dir=data_dir)
    data_logger.log("Min frequency: {}".format(experiment.min_freq))
    data_logger.log("Total German words in the vocabulary: {}".format(len(SRC.vocab)))
    data_logger.log("Total English words in the vocabulary: {}".format(len(TRG.vocab)))
    data_logger.log("")


def experiment_2x(data_logger):
    experiment = Experiment()
    experiment.corpus = "europarl"
    src_lang = experiment.src_lang = "de"
    experiment.trg_lang = "en"
    experiment.lang_code = "de"
    experiment.set_train(170000*2)
    experiment.set_val(1020*2)
    experiment.set_test(1190*2)
    experiment.voc_limit = VOC_LIMIT
    experiment.min_freq = MIN_FREQ

    SRC, TRG, _, _, _, _, _, _, _, _ = \
        get_vocabularies_iterators(src_lang, experiment, data_dir=data_dir)

    data_logger.log("Data information {} | DE > EN".format(experiment.train_samples))
    data_logger.log(
        "Total samples: {}".format(experiment.train_samples + experiment.val_samples + experiment.test_samples))
    data_logger.log("Total train: {} \t Total val: {} \t Total test: {}")
    data_logger.log("Min frequency: {}".format(experiment.min_freq))
    data_logger.log("Vocabulary limit: {}".format(experiment.voc_limit))

    data_logger.log("Total German words in the vocabulary: {}".format(len(SRC.vocab)))
    data_logger.log("Total English words in the vocabulary: {}".format(len(TRG.vocab)))

    data_logger.log("Total German words in the training dataset: {}".format(sum(SRC.vocab.freqs.values())))
    data_logger.log("Total English words in the training dataset: {}".format(sum(TRG.vocab.freqs.values())))

    #### Min freq 5
    experiment.min_freq = 5
    SRC, TRG, _, _, _, _, _, _, _, _ = \
        get_vocabularies_iterators(src_lang, experiment, data_dir=data_dir)
    data_logger.log("Min frequency: {}".format(experiment.min_freq))
    data_logger.log("Total German words in the vocabulary: {}".format(len(SRC.vocab)))
    data_logger.log("Total English words in the vocabulary: {}".format(len(TRG.vocab)))
    data_logger.log("")

def experiment_4x(data_logger):
    experiment = Experiment()
    experiment.corpus = "europarl"
    src_lang = experiment.src_lang = "de"
    experiment.trg_lang = "en"
    experiment.lang_code = "de"
    experiment.set_train(170000 * 4)
    experiment.set_val(1020 * 4)
    experiment.set_test(1190 * 4)
    experiment.voc_limit = VOC_LIMIT
    experiment.min_freq = MIN_FREQ

    SRC, TRG, _, _, _, _, _, _, _, _ = \
        get_vocabularies_iterators(src_lang, experiment, data_dir=data_dir)

    data_logger.log("Data information {} | DE > EN".format(experiment.train_samples))
    data_logger.log(
        "Total samples: {}".format(experiment.train_samples + experiment.val_samples + experiment.test_samples))
    data_logger.log("Total train: {} \t Total val: {} \t Total test: {}")
    data_logger.log("Min frequency: {}".format(experiment.min_freq))
    data_logger.log("Vocabulary limit: {}".format(experiment.voc_limit))

    data_logger.log("Total German words in the vocabulary: {}".format(len(SRC.vocab)))
    data_logger.log("Total English words in the vocabulary: {}".format(len(TRG.vocab)))

    data_logger.log("Total German words in the training dataset: {}".format(sum(SRC.vocab.freqs.values())))
    data_logger.log("Total English words in the training dataset: {}".format(sum(TRG.vocab.freqs.values())))
    data_logger.log("")


def experiment_full(data_logger):
    experiment = Experiment()
    experiment.corpus = "europarl"
    src_lang = experiment.src_lang = "de"
    experiment.trg_lang = "en"
    experiment.lang_code = "de"
    experiment.set_train(0)
    experiment.set_val(0)
    experiment.set_test(0)
    experiment.voc_limit = VOC_LIMIT
    experiment.min_freq = MIN_FREQ

    SRC, TRG, _, _, _, _, _, _, _, _ = \
        get_vocabularies_iterators(src_lang, experiment, data_dir=data_dir)

    data_logger.log("Data information {} | DE > EN".format(experiment.train_samples))
    data_logger.log(
        "Total samples: {}".format(experiment.train_samples + experiment.val_samples + experiment.test_samples))
    data_logger.log("Total train: {} \t Total val: {} \t Total test: {}")
    data_logger.log("Min frequency: {}".format(experiment.min_freq))
    data_logger.log("Vocabulary limit: {}".format(experiment.voc_limit))

    data_logger.log("Total German words in the vocabulary: {}".format(len(SRC.vocab)))
    data_logger.log("Total English words in the vocabulary: {}".format(len(TRG.vocab)))

    data_logger.log("Total German words in the training dataset: {}".format(sum(SRC.vocab.freqs.values())))
    data_logger.log("Total English words in the training dataset: {}".format(sum(TRG.vocab.freqs.values())))
    data_logger.log("")


if __name__ == '__main__':

    data_logger = Logger(path=data_dir, file_name="de-en.log")
    experiment_1x(data_logger)
    experiment_2x(data_logger)
    experiment_4x(data_logger)
    experiment_full(data_logger)
