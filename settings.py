import itertools
import os


##### Directories ###########
from project import get_full_path

ROOT = get_full_path(os.path.join("."))
DATA_DIR = get_full_path(os.path.join(ROOT, "data"))
DATA_DIR_RAW = get_full_path(os.path.join(DATA_DIR, "raw"))
DATA_DIR_PREPRO = get_full_path(os.path.join(DATA_DIR, "preprocessed"))

MAX_SUPPORTED_LENGTH = 50


##### Files ##############

RAW_TATOEBA = "tatoeba_deu-eng.txt"

#### Prefixes ######

TRAIN = "train"
VAL = "val"
TEST = "test"

PREFIXES = [TRAIN, VAL, TEST]

#### Suffixes ####
LANG1 = "en"
LANG2 = "de"

LANGUAGES = [LANG1, LANG2]


SPLITTINGS = list('.'.join(x) for x in list(itertools.product(PREFIXES, LANGUAGES)))


MODEL_STORE = get_full_path(ROOT, "results")


### spacy models ###

SPACY_EN = "en_core_web_sm"
SPACY_DE = "de_core_news_sm"

###python -m spacy download en_core_web_sm

RAW_EUROPARL = "raw_europarl.tsv"

DEFAULT_DEVICE = "cpu"