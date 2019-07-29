import os
from project import get_full_path

###### Directory settings

ROOT = get_full_path(os.path.join("."))
DATA_DIR = get_full_path(os.path.join(ROOT, "data"))
DATA_DIR_RAW = get_full_path(os.path.join(DATA_DIR, "raw"))
DATA_DIR_PREPRO = get_full_path(os.path.join(DATA_DIR, "preprocessed"))

MODEL_STORE = get_full_path(ROOT, "results")

RESULTS_DIR = get_full_path(ROOT, "results")


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
SUPPORTED_LANGS = dict({"en":"en_core_web_sm", "de":"de_core_news_sm", "it": "it_core_news_sm"})

#### Links to pretrained embeddings, see: https://fasttext.cc/docs/en/pretrained-vectors.html
PRETRAINED_URL_EN = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"
PRETRAINED_URL_LANG_CODE = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{}.300.vec.gz"
