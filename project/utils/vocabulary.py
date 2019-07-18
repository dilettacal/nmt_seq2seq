import gzip
import logging
import os
import shutil
import tarfile
import time
import zipfile
from urllib.request import urlretrieve
import torch
from torchtext import data, datasets, data as data
from torchtext.data import Field, Dataset
import urllib.request
import io
import gzip

from torchtext.utils import reporthook
from tqdm import tqdm

import project
from project import get_full_path
from project.utils.constants import SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN
from project.utils.data.download import maybe_download_and_extract
from project.utils.preprocessing import get_custom_tokenizer
from project.utils.utils import convert_time_unit
from settings import DATA_DIR_PREPRO, PRETRAINED_URL_EN, PRETRAINED_URL_LANG_CODE
import random
from torchtext import vocab

from settings import SEED
random.seed(SEED)


"""
This script contains the Seq2SeqDataset class to access bilingual text files
and other functions to create vocabularies and print some information 

"""

logger = logging.getLogger(__name__)
class EmbeddingLoader(vocab.Vectors):
    def __init__(self, name, cache=None,
                 url=None, unk_init=None, max_vectors=None):
        self.name = name
        super(EmbeddingLoader, self).__init__(name=name, cache=cache, url=url, unk_init=unk_init, max_vectors=max_vectors)

    @classmethod
    def download_embeddings(cls, url, file_name, cache, unk_init=None, max_vectors=None):
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=cache) as t:
            try:
                response = urlretrieve(url, cache, reporthook=reporthook(t))
            except KeyboardInterrupt as e:  # remove the partial zip file
                raise e

            compressed_file = io.BytesIO(response.read())
            decompressed_file = gzip.GzipFile(fileobj=compressed_file)
            file_path = os.path.join(cache, file_name)
            with open(file_path, 'wb') as outfile:
                outfile.write(decompressed_file.read())
        return cls(file_name, url=None, unk_init=unk_init, max_vectors=max_vectors)

class Seq2SeqDataset(Dataset):
    """
    Defines a dataset for machine translation.
    Part of this code is taken from the original source code TranslationDatset:
    See: https://github.com/pytorch/text/blob/master/torchtext/datasets/translation.py#L10

    """

    @staticmethod
    def sort_key(x):
        return (len(x.src), len(x.trg))

    def __init__(self, path, exts, fields, truncate=0, reduce=0):

        if not isinstance(fields[0], (tuple, list)):
           # print(fields)
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = self._generate_examples(src_path, trg_path, fields, truncate=truncate, reduce=reduce)

        super(Seq2SeqDataset, self).__init__(examples, fields)

    def _generate_examples(self, src_path, trg_path, fields, truncate, reduce):
        examples = []
        src_lines = [line.strip("\n") for line in
                     open(os.path.join(src_path), mode="r",
                          encoding="utf-8").readlines() if line]
        trg_lines = [line.strip("\n") for line in
                     open(os.path.join(trg_path), mode="r",
                          encoding="utf-8").readlines() if line]

        assert len(src_lines) == len(trg_lines)
        combined = list(zip(src_lines, trg_lines))
       # random.shuffle(combined)

        for i, (src_line, trg_line) in enumerate(combined):
            src_line, trg_line = src_line.strip(), trg_line.strip()
            if src_line != '' and trg_line != '':
                if truncate > 0:
                    src_line, trg_line = src_line.split(" "), trg_line.split(" ")
                    src_line = src_line[:truncate]
                    trg_line = trg_line[:truncate]
                    assert (len(src_line) <= truncate) and (len(trg_line) <= truncate)
                    src_line = ' '.join(src_line)
                    trg_line = ' '.join(trg_line)

                examples.append(data.Example.fromlist(
                    [src_line, trg_line], fields))

            if reduce > 0 and i == reduce:
                break

        return examples

    @classmethod
    def splits(cls, path=None, root='', train=None, validation=None,
               test=None, reduce = [0,0,0], reverse_input = False, **kwargs):

        exts = kwargs["exts"]
        reduce_samples = reduce
        fields = kwargs["fields"]
        truncate = kwargs.get("truncate", 0)
        if train or train != "":
            train_data = cls(os.path.join(path, train), exts=exts, reduce=reduce_samples[0], truncate=truncate, fields=fields)
        else: train_data = None

        if validation or validation != "":
            val_data = cls(os.path.join(path, validation), exts=exts, reduce=reduce_samples[1], truncate=truncate, fields=fields)
        else: val_data = None

        if test or test != "":
            test_data = cls(os.path.join(path, test), exts=exts, reduce=reduce_samples[2], truncate=truncate, fields=fields)
        else: test_data = None
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


def get_vocabularies_iterators(experiment, data_dir=None, max_len=30):
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

    char_level = experiment.char_level
    corpus = experiment.corpus
    language_code = experiment.lang_code
    reduce = experiment.reduce
    print("Vocabulary limit:",voc_limit)

    reverse_input = experiment.reverse_input
    print("Source reversed:", reverse_input)

    print("Required samples:")
    print(experiment.train_samples, experiment.val_samples, experiment.test_samples)

    src_vec, trg_vec = None, None

    ### Define tokenizers ####
    if char_level:
        src_tokenizer, trg_tokenizer = get_custom_tokenizer("en", "c"), get_custom_tokenizer("de", "c")
    else:
        ### equivalent to str.split(" ") but it considers punctuation and not only spaces as token boundaries. It also removes duplicate spaces.
        '''sent = "( DE ) Mr President , starting the agenda in this way with a brief debate on the Berlin statement is a good choice ."
           src_tokenizer.tokenize(sent) -->  
           ['(', 'DE', ')', 'Mr', 'President', ',', 'starting', 'the', 'agenda', 'in', 'this', 'way', 'with', 'a', 'brief', 'debate', 'on', 'the', 'Berlin', 'statement', 'is', 'a', 'good', 'choice', '.']
           
           sent.split(" ")
           ['(', 'DE', ')', 'Mr', 'President', ',', 'starting', 'the', 'agenda', 'in', 'this', 'way', 'with', 'a', 'brief', 'debate', 'on', 'the', 'Berlin', 'statement', 'is', 'a', 'good', 'choice', '.']
        '''
        spacy_pretok = True if corpus == "europarl" else False
        src_tokenizer, trg_tokenizer = get_custom_tokenizer("en", "w", spacy_pretok=spacy_pretok), get_custom_tokenizer("de", "w", spacy_pretok=spacy_pretok) #
        #### TODO: Rimuovi prima di consegnare :-)
        if spacy_pretok:
            assert isinstance(src_tokenizer, project.utils.preprocessing.SplitTokenizer)
            assert isinstance(trg_tokenizer, project.utils.preprocessing.SplitTokenizer)
        else:
            assert isinstance(src_tokenizer, project.utils.preprocessing.SpacyTokenizer)
            assert isinstance(trg_tokenizer, project.utils.preprocessing.SpacyTokenizer)

    if experiment.pretrained:
        ### retrieve word vectors
        embedding_dir = os.path.join(DATA_DIR_PREPRO, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)

        maybe_download_and_extract(download_dir=embedding_dir, url=PRETRAINED_URL_LANG_CODE.format(language_code),
                                   raw_file='cc.{}.300.vec'.format(language_code))
        maybe_download_and_extract(download_dir=embedding_dir, url=PRETRAINED_URL_EN,
                                   raw_file='cc.en.300.vec')
        init_unk = torch.Tensor.normal_
        if experiment.get_src_lang() == "en":
            src_vec = vocab.Vectors(name='cc.en.300.vec', cache=embedding_dir, unk_init = init_unk)
            trg_vec = vocab.Vectors(name='cc.{}.300.vec'.format(language_code), cache=embedding_dir, unk_init = init_unk)

        else:
            src_vec = vocab.Vectors('cc.{}.300.vec'.format(language_code), embedding_dir, unk_init = init_unk)
            trg_vec = vocab.Vectors('cc.en.300.vec', embedding_dir, unk_init = init_unk)

    src_vocab = Field(tokenize=lambda s: src_tokenizer.tokenize(s), include_lengths=False,init_token=None, eos_token=None, pad_token=PAD_TOKEN, unk_token=UNK_TOKEN, lower=True)
    trg_vocab = Field(tokenize=lambda s: trg_tokenizer.tokenize(s), include_lengths=False,init_token=SOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, unk_token=UNK_TOKEN, lower=True)
    print("Fields created!")

    ####### create splits ##########

    if corpus == "europarl":

        root = get_full_path(DATA_DIR_PREPRO)
        #print("Root:", root)
        if not data_dir:
            data_dir = os.path.join(root, corpus, language_code, "splits", str(max_len)) # local directory

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
        path = get_full_path(DATA_DIR_PREPRO, "iwslt")
        os.makedirs(path, exist_ok=True)
        exts = (".en", ".de") if experiment.get_src_lang() == "en" else (".de", ".en")
        train, val, test = datasets.IWSLT.splits(root=path,
                                                 exts=exts, fields=(src_vocab, trg_vocab),
                                                 filter_pred=lambda x: max(len(vars(x)['src']), len(vars(x)['trg'])) <= experiment.truncate)

        samples = None
        end = time.time()
        print("Duration: {}".format(convert_time_unit(end - start)))
        print("Total number of sentences: {}".format((len(train) + len(val) + len(test))))

    if voc_limit > 0:
        if experiment.pretrained:
            src_vocab.build_vocab(train, min_freq=min_freq, max_size=voc_limit, vectors=src_vec)
            trg_vocab.build_vocab(train, min_freq=min_freq, max_size=voc_limit, vectors=trg_vec)
        else:
            src_vocab.build_vocab(train, min_freq=min_freq, max_size=voc_limit)
            trg_vocab.build_vocab(train, min_freq=min_freq, max_size=voc_limit)
        print("Vocabularies created!")
    else:
        if experiment.pretrained:
            src_vocab.build_vocab(train, min_freq=min_freq, vectors=src_vec)
            trg_vocab.build_vocab(train, min_freq=min_freq, vectors=trg_vec)
        else:
            src_vocab.build_vocab(train, min_freq=min_freq)
            trg_vocab.build_vocab(train, min_freq=min_freq)
        print("Vocabularies created!")

    #print(trg_vocab.vocab.vectors[trg_vocab.vocab.stoi["the"]])
    #### Iterators #####

    # Create iterators to process text in batches of approx. the same length
    train_iter = data.BucketIterator(train, batch_size=experiment.batch_size, device=device, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), shuffle=True)
    val_iter = data.BucketIterator(val, 1, device=device, repeat=False, sort_key=lambda x: (len(x.src)), shuffle=True)
    test_iter = data.Iterator(test, batch_size=1, device=device, repeat=False, sort_key=lambda x: (len(x.src)), shuffle=False)

    if samples:
        samples_iter = data.Iterator(samples[0], batch_size=1, device=device, repeat=False, shuffle=False, sort_key=lambda x: (len(x.src)))
    else: samples_iter = None

    return src_vocab, trg_vocab, train_iter, val_iter, test_iter, train, val, test, samples, samples_iter


def print_data_info(logger, train_data, valid_data, test_data, src_field, trg_field, experiment):
    """ This prints some useful stuff about our data sets. """
    if experiment.corpus == "":
        corpus_name = "IWLST"
    else:
        corpus_name = experiment.corpus
    logger.log("Dataset in use: {}".format(corpus_name.upper()))

    logger.log("Data set sizes (number of sentence pairs):")
    logger.log('train {}'.format(len(train_data)))
    logger.log('valid {}'.format(len(valid_data)))
    logger.log('test {}'.format(len(test_data)))

    logger.log("First training example:")
    logger.log("src: {}".format(" ".join(vars(train_data[0])['src'])))
    logger.log("trg: {}".format(" ".join(vars(train_data[0])['trg'])))

    logger.log("Most common words (src):")
    logger.log("\n".join(["%20s %10d" % x for x in src_field.vocab.freqs.most_common(20)]))
    logger.log("Most common words (trg):")
    logger.log("\n".join(["%20s %10d" % x for x in trg_field.vocab.freqs.most_common(20)]))

    logger.log("Total UNKs in the source dataset: {}".format(src_field.vocab.freqs[UNK_TOKEN]))
    logger.log("Total UNKs in the target dataset: {}".format(trg_field.vocab.freqs[UNK_TOKEN]))

    logger.log("First 10 words (src):")
    logger.log("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(src_field.vocab.itos[:10])))
    logger.log("First 10 words (trg):")
    logger.log("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(trg_field.vocab.itos[:10])))

    logger.log("Number of Vocabulary source words (types): {}".format(len(src_field.vocab)))
    logger.log("Number of Vocabulary target words (types): {}".format(len(trg_field.vocab)))

    logger.log("Total SRC words in the training dataset: {}".format(sum(src_field.vocab.freqs.values())))
    logger.log("Total TRG words in the training dataset: {}".format(sum(trg_field.vocab.freqs.values())))

    logger.log("Minimal word frequency (src/trg): {}".format(experiment.min_freq))
