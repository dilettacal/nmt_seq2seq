import itertools
import os


##### Directories ###########
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



##### Files ##############

RAW_TATOEBA = "tatoeba_deu-eng.txt"

#### Prefixes ######

TRAIN = "train"
VAL = "val"
TEST = "test"

PREFIXES = [TRAIN, VAL, TEST]

#### Suffixes ####




### SPACY PART ###
### New languages should be written here

LANG1 = "en"
LANG2 = "de"

#### Model names ####
SPACY_EN = "en_core_web_sm"
SPACY_DE = "de_core_news_sm"

#### Update supported languages #####
SUPPORTED_LANGS = dict({LANG1:SPACY_EN, LANG2:SPACY_DE})


###python -m spacy download en_core_web_sm

TEST_TRANSLATIONS = [""] ### translations which are considered during the training phase, taken from the training set