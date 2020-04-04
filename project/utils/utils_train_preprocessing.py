import os
import time
from collections import Counter

from torchtext import datasets, data as data
from torchtext.data import Field
from project.utils.constants import PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN
from project.utils.get_tokenizer import get_custom_tokenizer
from project.utils.utils_functions import convert_time_unit
from project.utils.datasets import Seq2SeqDataset
from settings import DATA_DIR_PREPRO
import numpy as np


def get_vocabularies_and_iterators(experiment, data_dir=None, max_len=30):
    """
    Creates vocabularies and iterators for the experiment
    :param experiment: the Experiment object including all settings about the experiment
    :param data_dir: the directory where data is stored in. If None, default is applied
    :param max_len: the max length, default is the sentence max length considered during tokenization process
    :return: src vocabulary, trg vocabulary, datasets and iteratotrs + sample iterator if dataset europarl is used
    """

    device = experiment.get_device()

    #### Create torchtext fields
    ####### SRC, TRG
    voc_limit = experiment.voc_limit
    min_freq = experiment.min_freq

    corpus = experiment.corpus
    language_code = experiment.lang_code
    reduce = experiment.reduce
    print("Vocabulary limit:",voc_limit)

    reverse_input = experiment.reverse_input
    print("Source reversed:", reverse_input)

    print("Required samples:")
    print(experiment.train_samples, experiment.val_samples, experiment.test_samples)

    PREPRO = False if corpus == "europarl" else True
    MODE = "w"

    src_tokenizer, trg_tokenizer = get_custom_tokenizer("en", mode=MODE, prepro=PREPRO), get_custom_tokenizer(language_code, mode=MODE, prepro=PREPRO)

    src_vocab = Field(tokenize=lambda s: src_tokenizer.tokenize(s), include_lengths=False,init_token=None, eos_token=None, pad_token=PAD_TOKEN, unk_token=UNK_TOKEN, lower=True)
    trg_vocab = Field(tokenize=lambda s: trg_tokenizer.tokenize(s), include_lengths=False,init_token=SOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, unk_token=UNK_TOKEN, lower=True)
    print("Fields created!")

    ####### create splits ##########

    if corpus == "europarl":

        root = os.path.expanduser(DATA_DIR_PREPRO)
        if not data_dir:
            data_dir = os.path.join(root, corpus, language_code, "splits", str(max_len)) # local directory

        # check if files have been preprocessed
        try:
            files = os.listdir(data_dir)
            if len(files) < 8:
                print("ERROR: Not enough training files found at {}!\nTraining the model on the Europarl dataset requires train, val, test and samples splits for each language!".format(data_dir))
                print("Please drerun the script 'preprocess.py' for the given <lang_code>!")
        except FileNotFoundError:
            print("ERROR: Training files not found at {}!".format(data_dir))
            print("Please run the 'preprocess.py' script for the given <lang_code> before training the model!")
            exit(-1)

        print("Loading data...")
        start = time.time()
        file_type = experiment.tok
        exts = ("."+experiment.get_src_lang(), "."+experiment.get_trg_lang())
        train, val, test = Seq2SeqDataset.splits(fields=(src_vocab, trg_vocab),
                                                 exts=exts, train="train."+file_type, validation="val."+file_type, test="test."+file_type,
                                                 path=data_dir, reduce=reduce, truncate=experiment.truncate)

        ### samples is used to check translations during the training phase
        samples = Seq2SeqDataset.splits(fields=(src_vocab, trg_vocab), exts=exts,
                                        train="samples."+file_type,
                                        validation="", test="",
                                        path=data_dir)
        end = time.time()
        print("Duration: {}".format(convert_time_unit(end - start)))
        print("Total number of sentences: {}".format((len(train) + len(val) + len(test))))

    else:
        #### Training on IWSLT torchtext corpus #####
        print("Loading data...")
        start = time.time()
        path = os.path.expanduser(os.path.join(DATA_DIR_PREPRO, "iwslt"))
        os.makedirs(path, exist_ok=True)
        exts = (".en", ".de") if experiment.get_src_lang() == "en" else (".de", ".en")
        ## see: https://lukemelas.github.io/machine-translation.html
        train, val, test = datasets.IWSLT.splits(root=path,
                                                 exts=exts, fields=(src_vocab, trg_vocab),
                                                 filter_pred=lambda x: max(len(vars(x)['src']), len(vars(x)['trg'])) <= experiment.truncate)

        samples = None
        end = time.time()
        print("Duration: {}".format(convert_time_unit(end - start)))
        print("Total number of sentences: {}".format((len(train) + len(val) + len(test))))


    if voc_limit > 0:
        src_vocab.build_vocab(train, min_freq=min_freq, max_size=voc_limit)
        trg_vocab.build_vocab(train, min_freq=min_freq, max_size=voc_limit)
        print("Vocabularies created!")
    else:
        src_vocab.build_vocab(train, min_freq=min_freq)
        trg_vocab.build_vocab(train, min_freq=min_freq)
        print("Vocabularies created!")

    #### Iterators #####
    # Create iterators to process text in batches of approx. the same length
    train_iter = data.BucketIterator(train, batch_size=experiment.batch_size, device=device, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), shuffle=True)
    val_iter = data.BucketIterator(val, 1, device=device, repeat=False, sort_key=lambda x: (len(x.src)), shuffle=True)
    test_iter = data.Iterator(test, batch_size=1, device=device, repeat=False, sort_key=lambda x: (len(x.src)), shuffle=False)

    if samples[0].examples:
        samples_iter = data.Iterator(samples[0], batch_size=1, device=device, repeat=False, shuffle=False, sort_key=lambda x: (len(x.src)))
    else: samples_iter = None

    return src_vocab, trg_vocab, train_iter, val_iter, test_iter, train, val, test, samples, samples_iter


def print_info(logger, train_data, valid_data, test_data, val_iter, test_iter, src_field, trg_field, experiment):
    """ This prints some useful stuff about our data sets. """
    if experiment.corpus == "":
        corpus_name = "IWLST"
    else:
        corpus_name = experiment.corpus
    logger.log("Dataset in use: {}".format(corpus_name.upper()))

    logger.log("Data set sizes (number of sentence pairs):")
    logger.log('train {}'.format(len(train_data)-1))
    logger.log('valid {}'.format(len(valid_data)-1))
    logger.log('test {}'.format(len(test_data)-1))
    #length_checker(train_data, valid_data, test_data)

    logger.log("First training example:")
    logger.log("src: {}".format(" ".join(vars(train_data[0])['src'])))
    logger.log("trg: {}".format(" ".join(vars(train_data[0])['trg'])))

    logger.log("Most common words (src):")
    logger.log("\n".join(["%20s %10d" % x for x in src_field.vocab.freqs.most_common(20)]))
    logger.log("Most common words (trg):")
    logger.log("\n".join(["%20s %10d" % x for x in trg_field.vocab.freqs.most_common(20)]))

    logger.log("First 10 words (src):")
    logger.log("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(src_field.vocab.itos[:10])))
    logger.log("First 10 words (trg):")
    logger.log("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(trg_field.vocab.itos[:10])))

    logger.log("Maximal vocabulary size: {}".format(experiment.voc_limit))
    logger.log("Minimal word frequency (src/trg): {}".format(experiment.min_freq))
    logger.log("Number of Vocabulary source words (types): {}".format(len(src_field.vocab)))
    logger.log("Number of Vocabulary target words (types): {}".format(len(trg_field.vocab)))
    logger.log("")
    logger.log("###### Training dataset information #######")
    total_src_unk = {k: v for k, v in src_field.vocab.freqs.items() if v < experiment.min_freq}
    total_trg_unk = {k: v for k, v in trg_field.vocab.freqs.items() if v < experiment.min_freq}

    logger.log("Total UNKs in source vocabulary: {}".format(len(total_src_unk.keys())))
    logger.log("Total UNKs in target vocabulary: {}".format(len(total_trg_unk.keys())))

    logger.log("Total SRC words in the training dataset: {}".format(sum(src_field.vocab.freqs.values())))
    logger.log("Total TRG words in the training dataset: {}".format(sum(trg_field.vocab.freqs.values())))

    src_val_unks, trg_val_unks = count_unks(val_iter, src_field, trg_field)
    src_valid_words, trg_valid_words = count_words(val_iter, src_field, trg_field)
    src_test_unks, trg_test_unks = count_unks(test_iter, src_field, trg_field)
    src_test_words, trg_test_words = count_words(test_iter, src_field, trg_field)

    ### Validaiton Analysis ####
    logger.log("")
    logger.log("###### Validaiton dataset information #######")

    logger.log("Validaition total source UNKs: {} | Validaition total target UNKs: {} ".format(src_val_unks, trg_val_unks))
    logger.log("Total SRC words in the test dataset: {}".format(src_valid_words + src_val_unks))
    logger.log("Total TRG words in the test dataset: {}".format(trg_valid_words + trg_val_unks))

    ### Test dataset analysis ####
    logger.log("")
    logger.log("###### Test dataset information #######")
    logger.log("Test total source UNKs: {} | Test total target UNKs: {} ".format(src_test_unks, trg_test_unks))

    logger.log("Total SRC words in the test dataset: {}".format(src_test_words + src_test_unks))
    logger.log("Total TRG words in the test dataset: {}".format(trg_test_words + trg_test_unks))



def count_words(data_iter, src_vocab, trg_vocab):
    src_words, trg_words = 0, 0
    exclusions = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
    for batch in data_iter:
        src = batch.src
        trg = batch.trg

        vectorized_src = [src_vocab.vocab.itos[i] for i in src]
        src_word = [w for w in vectorized_src if w not in exclusions]

        vectorized_trg = [trg_vocab.vocab.itos[i] for i in trg]
        trg_word = [w for w in vectorized_trg if w not in exclusions]

        src_words += len(src_word)
        trg_words += len(trg_word)
    return src_words, trg_words

def count_unks(data_iter, src_vocab, trg_vocab):
    src_unks, trg_unks = 0, 0
    for batch in data_iter:
        src = batch.src
        trg = batch.trg

        vectorized_src = [src_vocab.vocab.itos[i] for i in src]
        unk_src = [w for w in vectorized_src if w == UNK_TOKEN]

        vectorized_trg = [trg_vocab.vocab.itos[i] for i in trg]
        unk_trg= [w for w in vectorized_trg if w == UNK_TOKEN]
        src_unks += len(unk_src)
        trg_unks += len(unk_trg)
    return src_unks, trg_unks




def length_checker(train_data, valid_data, test_data):
    from pprint import pprint
    ### train lengths ####

    all_train_src = [elem for elem in train_data.__getattr__("src")]
    all_train_trg = [elem for elem in train_data.__getattr__("trg")]

    all_train_src_lens = list(map(lambda x: len(x), all_train_src))
    all_train_trg_lens = list(map(lambda x: len(x), all_train_trg))

    train_src_len_counter = Counter(all_train_src_lens)
    train_trg_len_counter = Counter(all_train_trg_lens)

    print("Training data lengths:")
    print("German:")
    pprint(train_src_len_counter.most_common(10))
    print("English")
    pprint(train_trg_len_counter.most_common(10))

    ### validation lengths ####
    all_val_src = [elem for elem in valid_data.__getattr__("src")]
    all_val_trg = [elem for elem in valid_data.__getattr__("trg")]

    all_val_src_lens = list(map(lambda x: len(x), all_val_src))
    all_val_trg_lens = list(map(lambda x: len(x), all_val_trg))

    val_src_len_counter = Counter(all_val_src_lens)
    val_trg_len_counter = Counter(all_val_trg_lens)

    print("Validation data lenghts")
    print("German")
    pprint(val_src_len_counter.most_common(10))
    print("English")
    pprint(val_trg_len_counter.most_common(10))

    ### test lengths ####
    all_test_src = [elem for elem in test_data.__getattr__("src")]
    all_test_trg = [elem for elem in test_data.__getattr__("trg")]

    all_test_src_lens = list(map(lambda x: len(x), all_test_src))
    all_test_trg_lens = list(map(lambda x: len(x), all_test_trg))

    test_src_len_counter = Counter(all_test_src_lens)
    test_trg_len_counter = Counter(all_test_trg_lens)

    print("Test data lengths")
    print("German:")
    pprint(test_src_len_counter.most_common(10))
    print("English:")
    pprint(test_trg_len_counter.most_common(10))
