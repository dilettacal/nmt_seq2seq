"""
THis file contains all needed tokenizers for the preprocessing and training steps.
"""

import string
import re
from settings import SUPPORTED_LANGS
from project.utils.external.tmx_to_text import glom_urls

### Regex ###
space_before_punct = r'\s([?.!\'"](?:\s|$))'
before_apos = r"\s+(['])"
after_apos = r"(['])\s+([\w])"
BOUNDARY_REGEX = re.compile(r'\b|\Z')  # see tmx2corpus
TAG_REGEX = re.compile(r'<[^>]+>')


############### Tokenizers ################

class BaseSequenceTokenizer(object):
    def __init__(self, lang):
        self.lang = lang.lower()
        self.only_tokenize = True
        self.type = "standard"

    def _tokenize(self, text):
        raise NotImplementedError

    def tokenize(self, sequence):
        tokens = self._tokenize(sequence)
        return tokens

    def set_mode(self, only_tokenize=True):
        self.only_tokenize = only_tokenize

    def set_type(self, type="standard"):
        self.type = type

    def _clean_text(self, text):
        if isinstance(text, list):
            text = ' '.join(text)
        text = re.sub(space_before_punct, r"\1", text)
        text = re.sub(after_apos, r"\1\2", text)
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
        return [tok.text for tok in self.nlp.tokenizer(sequence)]

class FastTokenizer(BaseSequenceTokenizer):
    def __init__(self, lang):
        super(FastTokenizer, self).__init__(lang)

    def _tokenize(self, sequence):
        ## Tokenizer from https://github.com/amake/TMX2Corpus/blob/master/tokenizer.py#L45
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
def get_custom_tokenizer(lang, mode="w", prepro=True):
    """
    This function returns the tokenizer based on the configurations. The function is used either during the first preprocessing phase and during training time
    :param lang: the tokenizer language (relevant for spacy)
    :param mode: Char-based ("c") or Word-based ("w")
    :param prepro: True during the preprocessing step, False during training preprocessing
    :return: tokenizer
    """
    assert mode.lower() in ["c", "w"], "Please provide 'c' or 'w' as mode (char-level, word-level)."
    if prepro:
        mode = "w"
    if mode == "w" and prepro:
        return select_word_based_tokenizer(lang)
    elif not prepro:
        if mode == "c":
            return CharBasedTokenizer(lang)
        else:
            return SplitTokenizer(lang)


def select_word_based_tokenizer(lang):
    """
    This functions returns the SpacyTokenizer for the given language, if spaCy model is available.
    If not, it returns standard FastTokenizer
    :param lang: lang_code
    :return: a Spacy-Based Tokenizer or a FastTokenizer
    """
    if lang in SUPPORTED_LANGS.keys():
        try:
            import spacy
            nlp = spacy.load(SUPPORTED_LANGS[lang],
                             disable=["parser", "tagger", "textcat"])  # makes it faster
            tokenizer = SpacyTokenizer(lang, nlp)

        except OSError:
            print("Spacy model for language {} not found. Please install it.".format(lang))
            tokenizer = FastTokenizer(lang)
        except ImportError:
            print(
                "Spacy not installed or model for the requested language has not been downloaded.\nFast Tokenizer is used")
            tokenizer = FastTokenizer(lang)
        except Exception as e:
            print("Something went wrong: {}".format(e))
            tokenizer = FastTokenizer(lang)
    else:
        try:
            import spacy
            nlp = spacy.load("xx_ent_wiki_sm",  # model name for multi-languge models 'xx'
                             disable=["parser", "tagger", "textcat"])  # makes it faster
            tokenizer = SpacyTokenizer(lang, nlp)
        except OSError:
            print("Spacy model for language xx not installed.")
            tokenizer = FastTokenizer(lang)
        except ImportError:
            print(
                "Spacy not installed or model for the requested language has not been downloaded.\nFast Tokenizer is used")
            tokenizer = FastTokenizer(lang)
        except Exception as e:
            print("Something went wrong: {}".format(e))
            tokenizer = FastTokenizer(lang)
    return tokenizer