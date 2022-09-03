import os

###### Directory settings

ROOT = os.path.expanduser(os.path.join("."))
DATA_DIR = os.path.expanduser(os.path.join(ROOT,"data"))
DATA_DIR_RAW = os.path.expanduser(os.path.join(DATA_DIR, "raw"))
DATA_DIR_PREPRO = os.path.expanduser(os.path.join(DATA_DIR, "preprocessed"))
MODEL_STORE = os.path.expanduser(os.path.join(ROOT,"results"))


##### Experiment settings

DEFAULT_DEVICE = "cpu"
LSTM = "lstm"
GRU = "gru"
VALID_CELLS = [LSTM, GRU]
VALID_MODELS = ["custom", "s"]

SEED = 42

BEST_MODEL_PATH = ""


#### Suffixes ####


### SPACY PART ###
### New languages should be written here
#### Update supported languages #####
SUPPORTED_LANGS = dict({"en":"en_core_web_sm", "de":"de_core_news_sm", "xx":"xx_ent_wiki_sm"})

#### Links to pretrained embeddings, see: https://fasttext.cc/docs/en/pretrained-vectors.html
PRETRAINED_URL_EN = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"
PRETRAINED_URL_LANG_CODE = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{}.300.vec.gz"


# Corpora mappings for automatic downloading and preprocessing
CORPORA_LINKS_TMX = {
    "europarl": "http://opus.nlpl.eu/download.php?f=Europarl/v7/tmx/",
    "ted": "https://opus.nlpl.eu/download.php?f=TED2020/v1/tmx/",
    "wikipedia": "https://opus.nlpl.eu/download.php?f=Wikipedia/v1.0/tmx/",
    "tatoeba": "https://opus.nlpl.eu/download.php?f=Tatoeba/v2022-03-03/tmx/"
}
CORPORA_LINKS_TXT = {
    "europarl": "http://opus.nlpl.eu/download.php?f=Europarl/v7/moses/",
    "ted": "https://opus.nlpl.eu/download.php?f=TED2020/v1/moses/",
    "wikipedia": "https://opus.nlpl.eu/download.php?f=Wikipedia/v1.0/moses/",
    "tatoeba": "https://opus.nlpl.eu/download.php?f=Tatoeba/v2022-03-03/moses/"
}
