import abc
import os
import random
import string
from datetime import datetime
import re

import tokenizer ## from tmx2corpus!!!!!
from project.utils.mappings import ENG_CONTRACTIONS_MAP, UMLAUT_MAP
from project.utils.utils import Logger
from settings import DATA_DIR_PREPRO, SUPPORTED_LANGS, SEED
from tmx2corpus import Converter

### Regex ###
space_before_punct = r'\s([?.!"](?:\s|$))'
before_apos = r"\s+(['])"
after_apos = r"(['])\s+([\w])"

### from tmx2corpus "tokenizer.py"
BOUNDARY_REGEX = re.compile(r'\b|\Z')


########## Wrapper around tmx2corpus dependency ##############

class EmptyFilter(object):
    def filter(self, bitext):
        filtered_texts = list(filter(lambda item: item[1] or item[1] != "", bitext.items()))
        return bool(len(filtered_texts) == 2)


class MaxLenFilter(object):
    def __init__(self, length):
        self.len = length

    def filter(self, bitext):
        filtered_texts = list(filter(lambda item: len(item[1].split(" ")) <= self.len, bitext.items()))
        return bool(len(filtered_texts) == 2)  # both texts match the given predicate

class MinLenFilter(object):
    def __init__(self, length):
        self.len = length

    def filter(self, bitext):
        filtered_texts = list(filter(lambda item: len(item[1].split(" ")) >= self.len, bitext.items()))
        return bool(len(filtered_texts)==2) # both texts match the given predicate

class TMXTokenizer(tokenizer.Tokenizer):
    def __init__(self, lang):
        #self.custom_tokenizer = custom_tokenizer
        super(TMXTokenizer, self).__init__(lang.lower())

    def _tokenize(self, text):
        tokens = []
        i = 0
        for m in BOUNDARY_REGEX.finditer(text):
            tokens.append(text[i:m.start()])
            i = m.end()
        ### The tokenization may include too much spaces
        tokens = ' '.join(tokens)
        tokens = tokens.strip()
        ### remove possible duplicate spaces
        tokens = re.sub(' +', ' ', tokens)
        return tokens.split(" ")

########## Project custom tokenizers ###########

class BaseSequenceTokenizer(object):
    def __init__(self, lang):
        self.lang = lang
        self.only_tokenize = True
        self.type = "standard"

    def _tokenize(self, text):
        tokens = self._custom_tokenize(text)
        text = self._clean_text(' '.join(tokens))
        return text.split(" ")

    def tokenize(self, text):
        return self._custom_tokenize(text)

    @abc.abstractmethod
    def _custom_tokenize(self, text):
        pass

    def set_mode(self, only_tokenize=True):
        self.only_tokenize = only_tokenize

    def set_type(self, type="standard"):
        self.type = type

    def _clean_text(self, text):
        text = re.sub(space_before_punct, r"\1", text)
        text = re.sub(before_apos, r"\1", text)
        text = re.sub(after_apos, r"\1\2", text)
        if self.lang == "en":
            text = expand_contraction(text, ENG_CONTRACTIONS_MAP)
        elif self.lang == "de":
            text = expand_contraction(text, UMLAUT_MAP)
        text = cleanup_digits(text)
        return text

class CharBasedTokenizer(BaseSequenceTokenizer):

    def __init__(self, lang):
        super(CharBasedTokenizer, self).__init__(lang)
        self.type = "char"

    def _custom_tokenize(self, text):
        return list(text)

class SpacyTokenizer(BaseSequenceTokenizer):
    def __init__(self, lang, model):
        self.nlp = model
        super(SpacyTokenizer, self).__init__(lang)
        self.type = "spacy"

    def _custom_tokenize(self, text):
        doc = self.nlp(text)
        if self.only_tokenize:
            return [tok.text for tok in doc]
        else:
            ents = self.get_entities(doc)
            tokens = [tok.text for tok in doc]
            tokens = self.replace_text(tokens, ents)
            tokens = [token if token.isupper() else token.lower() for token in tokens]
        return tokens

    def get_entities(self, doc):
        text_ents = [(str(ent), "PERSON") for ent in doc.ents if ent.label_ in ["PER", "PERSON"]]
        return text_ents

    def replace_text(self, text, mapping):
        if isinstance(text, list):
            text = ' '.join(text)
        for ent in mapping:
              replacee = str(ent[0])
              replacer = str(ent[1])
              try:
                   text = text.replace(replacee, replacer)
              except:
                   pass

        return text.split(" ") if isinstance(text, str) else text


class StandardSplitTokenizer(BaseSequenceTokenizer):
    def _custom_tokenize(self, text):
        #### like for the TMXTokenizer
        tokens = []
        i = 0
        for m in BOUNDARY_REGEX.finditer(text):
            tokens.append(text[i:m.start()])
            i = m.end()
        ### The tokenization may include too much spaces
        tokens = ' '.join(tokens)
        tokens = tokens.strip()
        ### remove possible duplicate spaces
        tokens = re.sub(' +', ' ', tokens)
        return tokens.split(" ")

def get_custom_tokenizer(lang, mode, fast=False):
    assert mode.lower() in ["c", "w"], "Please provide 'c' or 'w' as mode (char-level, word-level)."
    tokenizer = None
    if mode == "c":
        tokenizer = CharBasedTokenizer(lang)
    else:
        if fast:
            tokenizer = StandardSplitTokenizer(lang)
        else:
            ## this may last more than 1 hour
            if lang in SUPPORTED_LANGS.keys():
                try:
                    import spacy
                    nlp = spacy.load(SUPPORTED_LANGS[lang], disable=["parser", "tagger", "textcat"]) #makes it faster
                    tokenizer = SpacyTokenizer(lang, nlp)
                except ImportError:
                    print("Spacy not installed. Standard tokenization is used")
                    tokenizer = StandardSplitTokenizer(lang)
    return tokenizer



class TMXConverter(Converter):
    def __init__(self, output):
        super().__init__(output)


class TXTConverter(Converter):
    def __init__(self, output):
        super().__init__(output)

    def convert(self, files:list):
        pass


def remove_adjacent_same_label(line):
    if isinstance(line, str):
        line = line.split(" ")
    # Remove adjacent duplicate labels
    toks = [line[i] for i in range(len(line)) if (i==0) or line[i] != line[i-1]]
    line = ' '.join(toks).strip()
    ### remove duplicate spaces
    line = re.sub(r"\s\s+", " ", line)
    return line.strip() # as string


def cleanup_digits(line):
    """
    Ex:
    Turchi Report [A5-0303/2001] and Linkohr Report (A5-0297/2001) - am 20. Juni 2019

    :param line:
    :return:
    """

    line = line.translate(str.maketrans('', '', string.punctuation))
    # Turchi Report A503032001 and Linkohr Report A502972001 am 20 Juni 2019
    line = line.strip()
    ### replace digits
    # Turchi Report A503032001 and Linkohr Report A502972001 am NUM Juni NUM
    nums = [n for n in re.split(r"\D+", line) if n]
    line = ' '.join([word if not word in nums else "NUM" for word in line.split(" ")])
    ### Clean up regulations
    ### A503032001 --> LAW
    line = re.sub(r'[a-zA-Z]+[0-9]+',"LAW", line)
    line = remove_adjacent_same_label(line)
    ## final string: Turchi Report LAW and Linkohr Report LAW am NUM Juni NUM
    return line


def expand_contraction(sentence, mapping):
    contractions_patterns = re.compile('({})'.format('|'.join(mapping.keys())), flags=re.IGNORECASE | re.DOTALL)

    def replace_text(t):
        txt = t.group(0)
        if txt.lower() in mapping.keys():
            return mapping[txt.lower()]

    expanded_sentence = contractions_patterns.sub(replace_text, sentence)
    return expanded_sentence


def split_data(src_sents, trg_sents, test_ratio=0.3, seed=SEED):
    assert len(src_sents) == len(trg_sents)
    data = list(zip(src_sents, trg_sents))

    num_samples = len(data)
    print("Total samples: ", num_samples)

    test_range = int(num_samples * test_ratio)  # test dataset 0.1
    train_range = num_samples - test_range
    random.seed(seed)  # 30
    random.shuffle(data)

    data_set = data[:train_range]
    val_set = data[train_range:]

    # create test set
    num_samples = len(data_set)
    test_range = int(num_samples * 0.1)
    train_range = num_samples - test_range

    train_set = data_set[:train_range]
    test_set = data_set[train_range:]

    print(len(test_set) + len(train_set) + len(val_set))

    train_set = list(zip(*train_set))
    val_set = list(zip(*val_set))
    test_set = list(zip(*test_set))

    return train_set, val_set, test_set



flatten = lambda l: [item for sublist in l for item in sublist]


def generate_splits_from_plain_text(root=os.path.join(DATA_DIR_PREPRO, "europarl"), language_code="de", max_len=30, filename="europarl"):
    FILES = sorted(["train.en", "val.en", "test.en",
             "train.{}".format(language_code), "val.{}".format(language_code), "test.{}".format(language_code)])

    store_path = os.path.join(DATA_DIR_PREPRO, "europarl", language_code, str(max_len))
    os.makedirs(store_path, exist_ok=True)
    files = os.listdir(store_path)
    files = sorted([file.lower() for file in files if
                    (file.endswith(".en") or file.endswith(".{}".format(language_code))) and (
                            file.split(".")[0] in ["train", "test", "val"])])

    if files == FILES:
        print("Files already splitted!")
    else:
        split_logger = Logger(store_path, file_name="split.log")
        split_logger.log("Splitting information - {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        en_file = filename+".en"
        trg_file = filename+".{}".format(language_code)
        path_to_en = os.path.join(root, language_code, en_file)
        path_to_de = os.path.join(root, language_code, trg_file)


def persist_txt(lines, store_path, file_name, exts):
    with open(os.path.join(store_path, file_name + exts[0]), mode="w", encoding="utf-8") as src_out_file,\
            open(os.path.join(store_path, file_name + exts[1]), mode="w", encoding="utf-8") as trg_out_file:
        if len(lines) == 2:
            lines = list(zip(lines[0], lines[1]))
            for src, trg in lines:
                src_out_file.write("{}\n".format(src))
                trg_out_file.write("{}\n".format(trg))


