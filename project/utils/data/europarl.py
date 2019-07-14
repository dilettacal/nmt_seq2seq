########################################################################
#
# Functions for downloading the Europarl data-set from the internet
# and loading it into memory. This data-set is used for translation
# between English and most European languages.
#
# http://opus.nlpl.eu/
#
# This script is a modified version of the file from Tensorflow Tutorials, available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License.
#
# Copyright 2018 by Magnus Erik Hvass Pedersen
#
########################################################################

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.


#data_dir = "data/europarl/" # original script
from project.utils.data import download
from settings import DATA_DIR_PREPRO

########################################################################
import os

########################################################################
from project import get_full_path
from settings import DATA_DIR_RAW

DATA_DIR = get_full_path(DATA_DIR_RAW, "europarl")
# Base-URL for the data-sets on the internet.
#data_url_statmt = "http://www.statmt.org/europarl/v7/" ### Tensorflow Tutorials use this link
data_url_opus = "http://opus.nlpl.eu/download.php?f=Europarl/v7/moses/" ## or v3 if version3 should be used
data_url_tmx_opus = "http://opus.nlpl.eu/download.php?f=Europarl/v7/tmx/"


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.




def maybe_download_and_extract_europarl(language_code="de", tmx=False):
    """
    Download and extract the Europarl data-set if the data-file doesn't
    already exist in data_dir. The data-set is for translating between
    English and the given language-code (e.g. 'da' for Danish, see the
    list of available language-codes above).
    """

    data_dir = os.path.join(DATA_DIR, language_code)
    os.makedirs(data_dir, exist_ok=True)


    if tmx:
        ##http://opus.nlpl.eu/download.php?f=Europarl/v7/tmx/de-en.tmx.gz
        url = data_url_tmx_opus + language_code + "-"+ "en"+ ".tmx"+".gz"
        try:
            raw_file = download.maybe_download_and_extract(url=url, download_dir=data_dir,language_code=language_code)
            return raw_file
        except:
            ##http://opus.nlpl.eu/download.php?f=Europarl/v7/tmx/en-fr.tmx.gz
            url = data_url_tmx_opus + "en" + "-" + language_code + ".tmx"+".gz"
            raw_file = download.maybe_download_and_extract(url=url, download_dir=data_dir, language_code=language_code)
            return raw_file
    else:

        ## http://opus.nlpl.eu/download.php?f=Europarl/v7/moses/de-en.txt.zip
        url = data_url_opus + language_code + "-" + "en"+".txt"+".zip"
        try:
            download.maybe_download_and_extract(url=url, download_dir=data_dir, language_code=language_code)
        except:
            ## or: http://opus.nlpl.eu/download.php?f=Europarl/v7/moses/en-fr.txt.zip
            url = data_url_opus + "en" + "-" + language_code+".txt"+".zip"
            download.maybe_download_and_extract(url=url, download_dir=data_dir, language_code=language_code)


def load_data(english=True, language_code="da", start="", end="", tmx=False):
    """
    Load the data-file for either the English-language texts or
    for the other language (e.g. "da" for Danish).

    All lines of the data-file are returned as a list of strings.

    :param english:
      Boolean whether to load the data-file for
      English (True) or the other language (False).

    :param language_code:
      Two-char code for the other language e.g. "da" for Danish.
      See list of available codes above.

    :param start:
      Prepend each line with this text e.g. "ssss " to indicate start of line.

    :param end:
      Append each line with this text e.g. " eeee" to indicate end of line.

    :return:
      List of strings with all the lines of the data-file.
    """

    suffixes = ["en", language_code]

    print("Trying to load data...")

    if tmx:
        files = [file for file in os.listdir(os.path.join(DATA_DIR_PREPRO, "europarl", language_code)) if file.startswith("bitext.tok") and file.split(".")[-1] in suffixes]
        print(files)
        # Full path for the data-file.
        data_dir = os.path.join(DATA_DIR_PREPRO, "europarl", language_code)
    else:
        files = [file for file in os.listdir(os.path.join(DATA_DIR, language_code)) if file.startswith("Europarl.") and file.split(".")[-1] in suffixes]
        # Full path for the data-file.
        data_dir = os.path.join(DATA_DIR, language_code)
    if english:
        # Load the English data.
       # filename = "europarl-v7.{0}-en.en".format(language_code)
        filename = [file for file in files if file.endswith("en")][0]
        print(filename)
    else:
        # Load the other language.
       #filename = "europarl-v7.{0}-en.{0}".format(language_code)
        filename = [file for file in files if file.endswith(language_code)][0]


    path = os.path.join(data_dir, filename)

    # Open and read all the contents of the data-file.
    with open(path, encoding="utf-8") as file:
        # Read the line from file, strip leading and trailing whitespace,
        # prepend the start-text and append the end-text.
        texts = [start + line.strip() + end for line in file]

    return texts


########################################################################

if __name__ == '__main__':
    #TODO: gz not working
    maybe_download_and_extract_europarl(language_code="it", tmx=True)