import string
import re
from settings import SUPPORTED_LANGS
from project.utils.external.tmx_to_text import glom_urls

### Regex ###
space_before_punct = r'\s([?.!\'"](?:\s|$))'
before_apos = r"\s+(['])"
after_apos = r"(['])\s+([\w])"
BOUNDARY_REGEX = re.compile(r'\b|\Z')  #
TAG_REGEX = re.compile(r'<[^>]+>')


############### Tokenizers ################

class BaseSequenceTokenizer(object):
    def __init__(self, lang):
        self.lang = lang.lower()
        self.only_tokenize = True
        self.type = "standard"

    def _tokenize(self, text):
        '''Override this to implement the actual tokenization: Take string,
                return list of tokens.'''
        raise NotImplementedError

    def tokenize(self, sequence):
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
        super(FastTokenizer, self).__init__(lang)

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
def get_custom_tokenizer(lang, mode="w", spacy_pretok=False, prepro=True):
    """
    This function returns the tokenizer based on the configurations. The function is used either during the first preprocessing phase and during training time
    :param lang: the tokenizer language (relevant for spacy)
    :param mode: Char-based ("c") or Word-based ("w")
    :param spacy_pretok:
    :param prepro:
    :return:
    """
    assert mode.lower() in ["c", "w"], "Please provide 'c' or 'w' as mode (char-level, word-level)."
    tokenizer = None
    if mode == "c" and spacy_pretok or not prepro:
        tokenizer = CharBasedTokenizer(lang)
    elif mode == "c" and prepro:
        tokenizer = FastTokenizer(lang)
    elif mode == "w":
        if spacy_pretok:
            # if spacy tokenization already performed, then sentences can be splitted on spaces
            tokenizer = SplitTokenizer(lang)
        else:
            if not prepro:
                tokenizer = FastTokenizer(lang)
            else:
                if lang in SUPPORTED_LANGS.keys():
                    try:
                        import spacy
                        nlp = spacy.load(SUPPORTED_LANGS[lang],
                                         disable=["parser", "tagger", "textcat"])  # makes it faster
                        tokenizer = SpacyTokenizer(lang, nlp)
                    except ImportError or Exception:
                        print(
                            "Spacy not installed or model for the requested language has not been downloaded.\nFast Tokenizer is used")
                        tokenizer = FastTokenizer(lang)
                else:
                    try:
                        import spacy
                        nlp = spacy.load("xx",
                                         disable=["parser", "tagger", "textcat"])  # makes it faster
                        tokenizer = SpacyTokenizer(lang, nlp)
                    except ImportError or Exception:
                        print(
                            "Spacy not installed or model for the requested language has not been downloaded.\nFast Tokenizer is used")
                        tokenizer = FastTokenizer(lang)
                    #tokenizer = FastTokenizer(lang)

    return tokenizer


#### other tokenization utilities ###

def remove_adjacent_same_label(line):
    if isinstance(line, str):
        line = line.split(" ")
    # Remove adjacent duplicate labels
    toks = [line[i] for i in range(len(line)) if (i == 0) or line[i] != line[i - 1]]
    line = ' '.join(toks).strip()
    ### remove duplicate spaces
    line = re.sub(r"\s\s+", " ", line)
    return line.strip()  # as string


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
    line = re.sub(r'[a-zA-Z]+[0-9]+', "LAW", line)
    line = remove_adjacent_same_label(line)
    ## final string: Turchi Report LAW and Linkohr Report LAW am NUM Juni NUM
    return line



