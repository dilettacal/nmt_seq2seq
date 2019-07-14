import abc
import logging
import os
import random
import string
import time
import re

from project.utils.data.europarl import maybe_download_and_extract_europarl
from project.utils.mappings import ENG_CONTRACTIONS_MAP, UMLAUT_MAP
from project.utils.utils import convert, Logger
from settings import DATA_DIR_PREPRO, SUPPORTED_LANGS, SEED, DATA_DIR_RAW
from project.utils.tmx2corpus.tokenizer import Tokenizer, glom_urls
from project.utils.tmx2corpus.tmx2corpus import Converter, FileOutput, extract_tmx
from nltk.tokenize import RegexpTokenizer, PunktSentenceTokenizer

### Regex ###
space_before_punct = r'\s([?.!\'"](?:\s|$))'
before_apos = r"\s+(['])"
after_apos = r"(['])\s+([\w])"
BOUNDARY_REGEX = re.compile(r'\b|\Z')#
TAG_REGEX = re.compile(r'<[^>]+>')

PATTERN = r'\W+'

############# Raw file convertern ####



class TMXConverter(Converter):
    def __init__(self, output):
        super().__init__(output)
        self.tokenizers = {}

    def convert(self, files):
        self.suppress_count = 0
        self.output_lines = 0
        for tmx in files:
            print('Extracting %s' % os.path.basename(tmx))
            for bitext in extract_tmx(tmx):
                self.__output(bitext)
        logging.info('Output %d pairs', self.output_lines)
        if self.suppress_count:
            logging.info('Suppressed %d pairs', self.suppress_count)

    def __output(self, bitext):
        for fltr in self.filters:
            if not fltr.filter(bitext):
                self.suppress_count += 1
                return

        if bool(self.tokenizers):
            for lang, text in list(bitext.items()):
                tokenizer = self.tokenizers.get(lang, FastTokenizer(lang))
                ### modified: our tokenizers return tokens
                bitext['tok.' + lang] = ' '.join(tokenizer.tokenize(text))

        for lang in bitext.keys():
            self.output.init(lang)

        for lang, text in bitext.items():
            self.output.write(lang, text)

        self.output_lines += 1


############### Tokenizers ################


class BaseSequenceTokenizer(Tokenizer):
    def __init__(self, lang):
        super(BaseSequenceTokenizer, self).__init__(lang.lower())
        self.lang = lang
        self.only_tokenize = True
        self.type = "standard"

    def tokenize(self, sequence):
        if self.lang == "en":
            sequence = expand_contraction(sequence, ENG_CONTRACTIONS_MAP)
        elif self.lang == "de":
            sequence = expand_contraction(sequence, UMLAUT_MAP)
        tokens = self._tokenize(sequence)
       # return ' '.join(tokens)
        return tokens

    def set_mode(self, only_tokenize=True):
        self.only_tokenize = only_tokenize

    def set_type(self, type="standard"):
        self.type = type

    def _clean_text(self, text):
        if isinstance(text, list):
            text = ' '.join(text)
        text = re.sub(space_before_punct, r"\1", text)
      #  text = re.sub(before_apos, r"\1", text)
        text = re.sub(after_apos, r"\1\2", text)
        if self.lang == "en":
            text = expand_contraction(text, ENG_CONTRACTIONS_MAP)
        elif self.lang == "de":
            text = expand_contraction(text, UMLAUT_MAP)
       # text = cleanup_digits(text)
        return text

class CharBasedTokenizer(BaseSequenceTokenizer):

    def __init__(self, lang):
        super(CharBasedTokenizer, self).__init__(lang)
        self.type = "char"

    def tokenize(self, sequence):
        return self._tokenize(sequence)

    def _tokenize(self, text):
        return list(text)

class SpacyTokenizer(BaseSequenceTokenizer):
    def __init__(self, lang, model):
        self.nlp = model
        super(SpacyTokenizer, self).__init__(lang)
        self.type = "spacy"
        self.only_tokenize = True

    def _tokenize(self, sequence):
        if self.only_tokenize:
           # doc = self.nlp(sequence)
            return [tok.text for tok in self.nlp.tokenizer(sequence)]
        else:
            ### this takes really long ###
            sequence = cleanup_digits(sequence)
            doc = self.nlp(sequence)
            ents = self.get_entities(doc)
            tokens = [tok.text for tok in doc]
            tokens = self.replace_text(tokens, ents)
            tokens = [token if token.isupper() else token.lower() for token in tokens]
            tokens = self._clean_text(tokens)
            tokens = tokens.split(" ")
        return tokens


    ##### this could improve tokenization, not used in the project

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

class FastTokenizer(BaseSequenceTokenizer):
    def __init__(self, lang):
        super(BaseSequenceTokenizer, self).__init__(lang)

    def _tokenize(self, sequence):
        text = TAG_REGEX.sub('', sequence)
        text = re.sub(r"\s\s+", " ", text)
        tokens = []
        i = 0
        for m in BOUNDARY_REGEX.finditer(text):
            tokens.append(text[i:m.start()])
            i = m.end()
        if '://' in text or '@' in text:
            tokens = glom_urls(tokens)
        tokens = [tok for tok in tokens if not tok.strip() == '']
        return ' '.join(tokens).split(" ")

class SplitTokenizer(BaseSequenceTokenizer):

    def _tokenize(self, text):
        text = re.sub(r"\s\s+", " ", text)
        return text.split(" ")


##### Factory method ########
def get_custom_tokenizer(lang, mode, fast=False, spacy_pretok=True):
    assert mode.lower() in ["c", "w"], "Please provide 'c' or 'w' as mode (char-level, word-level)."
    tokenizer = None
    if mode == "c":
        tokenizer = CharBasedTokenizer(lang)
    else:
        if fast:
            tokenizer = FastTokenizer(lang)
        elif spacy_pretok:
            tokenizer = SplitTokenizer(lang)
        else:
            ## this may last more than 1 hour
            if lang in SUPPORTED_LANGS.keys():
                try:
                    import spacy
                    nlp = spacy.load(SUPPORTED_LANGS[lang], disable=["parser", "tagger", "textcat"]) #makes it faster
                    tokenizer = SpacyTokenizer(lang, nlp)
                except ImportError or Exception:
                    print("Spacy not installed or model for the requested language has not been downloaded.\nStandard tokenizer is used")
                    tokenizer = FastTokenizer(lang)
                    tokenizer.set_mode(True)
    return tokenizer



#### other tokenization utilities ###

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


##### Generates splits from the main dataset ####

def split_data(src_sents, trg_sents, val_ratio=0.1, train_ratio=0.8, seed=SEED):

    """
    Split the source and target sentences using the provided ratios and seed
    Default: 80, 10, 10
    :param src_sents: list containing only the source sentences
    :param trg_sents: list containing only the target sentences
    :param val_ratio: validation ratio
    :param train_ratio: training ration
    :param seed: splits on the provided feed, default see settings.py
    :return: splits
    """

    assert len(src_sents) == len(trg_sents)
    data = list(zip(src_sents, trg_sents))

    num_samples = len(data)
    print("Total samples: ", num_samples)


    print("Shuffling data....")
    random.seed(seed)  # 30
    random.shuffle(data)

    train_end = int(train_ratio*num_samples)
    validate_end = int(val_ratio*num_samples) + train_end
    train_set = data[:train_end]
    val_set = data[train_end:validate_end]
    test_set = data[validate_end:]
    print("Total train:", len(train_set))
    print("Total validation:", len(val_set))
    print("Total test:", len(test_set))
    print("All togheter:", len(test_set) + len(train_set) + len(val_set))

    samples = train_set[:5] + val_set[:5] + test_set[:5]

    train_set = list(zip(*train_set))
    val_set = list(zip(*val_set))
    test_set = list(zip(*test_set))

    samples_set = list(zip(*samples))
    return train_set, val_set, test_set, samples_set


def persist_txt(lines, store_path, file_name, exts):
    """
    Stores the given lines
    :param lines: bilingual list of sentences
    :param store_path: path to store the file in
    :param file_name:
    :param exts: tuple containing the extensions, should match the line order, default: (.en, lang_code)
    :return:
    """
    with open(os.path.join(store_path, file_name + exts[0]), mode="w", encoding="utf-8") as src_out_file,\
            open(os.path.join(store_path, file_name + exts[1]), mode="w", encoding="utf-8") as trg_out_file:
        if len(lines) == 2:
            lines = list(zip(lines[0], lines[1]))
            for src, trg in lines:
                src_out_file.write("{}\n".format(src))
                trg_out_file.write("{}\n".format(trg))


#### raw file preprocessing
def raw_preprocess(parser):
    #### preprocessing pipeline for tmx files
    ### download the files #####
    maybe_download_and_extract_europarl(language_code=parser.lang_code, tmx=True)
    corpus_name = parser.corpus
    lang_code = parser.lang_code
    file_type = parser.type
    path_to_raw_file = parser.path
    max_len, min_len = parser.max_len, parser.min_len

    COMPLETE_PATH = os.path.join(path_to_raw_file, parser.file)

    STORE_PATH = os.path.join(os.path.expanduser(DATA_DIR_PREPRO), corpus_name, lang_code, "splits", str(max_len))
    os.makedirs(STORE_PATH, exist_ok=True)

    assert file_type in ["tmx", "txt"]

    if file_type == "tmx":
        start = time.time()
        output_file_path = os.path.join(DATA_DIR_PREPRO, corpus_name, lang_code)
        files = [file for file in os.listdir(output_file_path) if
                 file.startswith("bitext.tok") or file.startswith("bitext.tok")]
        if len(files) >= 2:
            print("TMX file already preprocessd!")
        else:
            ### This conversion uses standard tokenizers, which splits sentences on spaces and punctuation, this is very fast
            converter = TMXConverter(output=FileOutput(output_file_path))
            ### tokenization by default with spacy, if spacy not available use standard tokenizer by tmx2corp
           # src_tokenizer, trg_tokenizer = get_custom_tokenizer("en", "w", spacy_pretok=False), get_custom_tokenizer(
            #    "de", "w", spacy_pretok=False)  # spacy is used
            #tokenizers = [src_tokenizer, trg_tokenizer]
            #converter.add_tokenizers(tokenizers)
            #converter.add_tokenizers()
            converter.convert([COMPLETE_PATH])
            print("Converted lines:", converter.output_lines)

        target_file = "bitext.{}".format(lang_code)
        src_lines = [line.strip("\n") for line in
                     open(os.path.join(output_file_path, "bitext.en"), mode="r",
                          encoding="utf-8").readlines() if line]
        trg_lines = [line.strip("\n") for line in
                     open(os.path.join(output_file_path, target_file), mode="r",
                          encoding="utf-8").readlines() if line]

        ### tokenize lines ####

        src_tokenizer, trg_tokenizer = get_custom_tokenizer("en", "w", spacy_pretok=False), get_custom_tokenizer(   "de", "w", spacy_pretok=False)  # spacy is used
        src_logger = Logger(output_file_path, file_name="bitext.tok.en")
        trg_logger = Logger(output_file_path, file_name="bitext.tok.{}".format(lang_code))

        temp_src_toks, temp_trg_toks = [], []

        with src_tokenizer.nlp.disable_pipes('tagger', 'parser', 'ner'):
            for i, doc in enumerate(src_tokenizer.nlp.pipe(src_lines, batch_size=1000)):
                tok_doc = ' '.join([tok.text for tok in doc])
                temp_src_toks.append(tok_doc)
                src_logger.log(tok_doc, stdout=True if i % 10000 ==0 else False)


        with trg_tokenizer.nlp.disable_pipes('tagger', 'parser', 'ner'):
            for i, doc in enumerate(trg_tokenizer.nlp.pipe(src_lines, batch_size=1000)):
                tok_doc = ' '.join([tok.text for tok in doc])
                temp_trg_toks.append(tok_doc)
                trg_logger.log(tok_doc, stdout=True if i % 10000 == 0 else False)

        if max_len > 0:
            files = ['.'.join(file.split(".")[:2]) for file in os.listdir(STORE_PATH) if
                     file.endswith("tok.en") or file.endswith("tok." + lang_code)]
            filtered_src_lines, filtered_trg_lines = [], []
            if files:
                print("File already reduced by length!")
            else:
                print("Filtering by length...")
                filtered_src_lines, filtered_trg_lines = [], []
                for src_l, trg_l in zip(temp_src_toks, temp_trg_toks):
                    src_l_s = src_l.strip()
                    trg_l_s = trg_l.strip()
                    ### remove possible duplicate spaces
                    src_l_s = re.sub(' +', ' ', src_l_s)
                    trg_l_s = re.sub(' +', ' ', trg_l_s)
                    if src_l_s != "" and trg_l_s != "":
                        src_l_spl, trg_l_spl = src_l_s.split(" "), trg_l_s.split(" ")
                        if len(src_l_spl) >= min_len and len(trg_l_spl) >= min_len:
                            if len(src_l_spl) <= max_len and len(trg_l_spl) <= max_len:
                                filtered_src_lines.append(' '.join(src_l_spl))
                                filtered_trg_lines.append(' '.join(trg_l_spl))
                assert len(filtered_src_lines) == len(filtered_trg_lines)

            src_lines, trg_lines = filtered_src_lines, filtered_trg_lines
            print("Splitting files...")
            train_data, val_data, test_data, samples_data = split_data(src_lines, trg_lines)
            persist_txt(train_data, STORE_PATH, "train.tok", exts=(".en", "." + lang_code))
            persist_txt(val_data, STORE_PATH, "val.tok", exts=(".en", "." + lang_code))
            persist_txt(test_data, STORE_PATH, "test.tok", exts=(".en", "." + lang_code))
            print("Generating samples files...")
            persist_txt(samples_data, STORE_PATH, file_name="samples.tok", exts=(".en", "." + lang_code))

        print("Total time:", convert(time.time() - start))
    else:
        # TODO
        pass