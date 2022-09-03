########################################################################
# Functions for downloading the Europarl data-set from the internet
# and loading it into memory. This data-set is used for translation
# between English and most European languages.
#
# http://opus.nlpl.eu/
#
# This script is a modified version of the file from Tensorflow Tutorials, available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
# Published under the MIT License.
# Copyright 2018 by Magnus Erik Hvass Pedersen
########################################################################

from project.utils.external import download
from settings import DATA_DIR_PREPRO

########################################################################
import os

########################################################################
from settings import DATA_DIR_RAW

DATA_DIR = os.path.expanduser(os.path.join(DATA_DIR_RAW, "{}"))

# Base-URL for the data-sets on the internet.
# Modified for handling tmx files

CORPORA_LINKS_TMX = {
    "europarl": "http://opus.nlpl.eu/download.php?f=Europarl/v7/tmx/",
    "ted": "https://opus.nlpl.eu/download.php?f=TED2020/v1/tmx/"
}

CORPORA_LINKS_TXT = {
    "europarl": "http://opus.nlpl.eu/download.php?f=Europarl/v7/moses/",
    "ted": "https://opus.nlpl.eu/download.php?f=TED2020/v1/moses/"
}
corpus_name = ""
data_url_opus = "http://opus.nlpl.eu/download.php?f={}/v7/moses/" ## or v3 if version3 should be used
data_url_tmx_opus = "http://opus.nlpl.eu/download.php?f={}/v7/tmx/"


def maybe_download_and_extract_corpus_from_opus(corpus = "Europarl", language_code="de", tmx=False):
    """
    Download and extract the opus corpus data-set if the data-file doesn't
    already exist in data_dir. The data-set is for translating between
    English and the given language-code (e.g. de)
    """
    assert corpus.lower() in CORPORA_LINKS_TMX.keys() or corpus.lower() in CORPORA_LINKS_TXT.keys(), \
        f"Please provide valid corpus name: {CORPORA_LINKS_TMX.keys()}"
    corpus_name = corpus.lower()
    data_dir = os.path.join(DATA_DIR.format(corpus_name), language_code)
    os.makedirs(data_dir, exist_ok=True)

    if tmx:
        ##http://opus.nlpl.eu/download.php?f=Europarl/v7/tmx/de-en.tmx.gz
        url = CORPORA_LINKS_TMX.get(corpus_name) + language_code + "-"+ "en"+ ".tmx"+".gz"
        try:
            raw_file = download.maybe_download_and_extract(url=url, download_dir=data_dir, language_code=language_code)
            return raw_file
        except:
            ##http://opus.nlpl.eu/download.php?f=Europarl/v7/tmx/en-fr.tmx.gz
            url = data_url_tmx_opus + "en" + "-" + language_code + ".tmx"+".gz"
            raw_file = download.maybe_download_and_extract(url=url, download_dir=data_dir, language_code=language_code)
            return raw_file
    else:
        ## http://opus.nlpl.eu/download.php?f=Europarl/v7/moses/de-en.txt.zip
        url = CORPORA_LINKS_TXT.get(corpus_name) + language_code + "-" + "en"+".txt"+".zip"
        try:
            download.maybe_download_and_extract(url=url, download_dir=data_dir, language_code=language_code)
        except:
            ## or: http://opus.nlpl.eu/download.php?f=Europarl/v7/moses/en-fr.txt.zip
            url = data_url_opus + "en" + "-" + language_code+".txt"+".zip"
            download.maybe_download_and_extract(url=url, download_dir=data_dir, language_code=language_code)



if __name__ == '__main__':
    maybe_download_and_extract_corpus_from_opus(corpus="ted", language_code="de", tmx=True)