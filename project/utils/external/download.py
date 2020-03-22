########################################################################
# Functions for downloading and extracting data-files from the internet.
# This file is part of the TensorFlow Tutorials available at:
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################
import gzip
import shutil
import sys
import os
import urllib.request
import tarfile
import zipfile
from abc import ABC, abstractmethod
from urllib.error import ContentTooShortError

from project.utils.external.europarl import GENERAL_DATA_DIR
from project.utils.utils_parsers import DatasetConfigParser
from settings import DATA_DIR_RAW


class CorpusDownloader(ABC):
    def __init__(self, dataset_config: DatasetConfigParser, corpus_name, lang_code="de"):
        self.dataset_config = dataset_config
        self.lang_code = lang_code
        self.datasets = self.read_sections()
        self.tmx = True if self.datasets.get("format") == "tmx" else False
        assert self.datasets.get(corpus_name, None) is not None
        self.dataset_name = self.datasets.get(corpus_name)
        self.download_dir = os.path.join(DATA_DIR_RAW, self.datasets.get(corpus_name))

    def read_sections(self):
        datasets = {}
        for section in self.dataset_config.sections:
            datasets[section] = self.dataset_config.read_section(section)
        return datasets

    @abstractmethod
    def download_files(self, base_url):
        filename = base_url.split('/')[-1]
        file_path = os.path.join(self.download_dir, filename)  # e.g. ./data/raw/europarl/de/en-de.tmx.gz
        if not os.path.exists(file_path):
            # Check if the download directory exists, otherwise create it.
            if not os.path.exists(self.download_dir):
                os.makedirs(self.download_dir)

            # Download the file from the internet.
            try:
                file_path, _ = urllib.request.urlretrieve(url=base_url,
                                                          filename=file_path,
                                                          reporthook=_print_download_progress)

                print("\nFile downloaded in:", file_path)  # ./data/raw/europarl/de/en-de.tmx.gz

                print()
                print("Download finished. Extracting files.")
                return file_path
            except ContentTooShortError as e:
                print(e)
                return False
        else:
            print("Data already downloaded and unpacked.")
            return False  # filename

    @abstractmethod
    def maybe_download_and_extract(self, url, raw_file=""):
        file_path = self.download_files(url)
        if file_path:
            if file_path.endswith(".zip"):
                zipfile.ZipFile(file=file_path, mode="r").extractall(self.download_dir)
                return True
            elif file_path.endswith((".tar.gz", ".tgz")):
                tarfile.open(name=file_path, mode="r:gz").extractall(self.download_dir)
                return True
            elif file_path.endswith(".gz"):
                # Modified for tmx files #
                raw_file = raw_file.format(self.lang_code)  # e.g. de-en.tmx
                download_dir = os.path.join(self.download_dir, raw_file)
                with gzip.open(file_path, 'rb') as gz:
                    with open(download_dir, 'wb') as uncompressed:
                        shutil.copyfileobj(gz, uncompressed)
                print("Done.")
                return raw_file


class EuroparlDatasetDownloader(CorpusDownloader):
    def __init__(self, dataset_config: DatasetConfigParser, corpus_name):
        super().__init__(dataset_config, corpus_name)

    def maybe_download_and_extract(self, url, raw_file="{}-en.tmx"):
        download = super().maybe_download_and_extract(url, raw_file)
        if download:
            # dataset is dict like {'name': 'europarl', 'genre': 'law', 'version': 'v7', 'url': 'http://opus.nlpl.eu/download.php?f=Europarl/v7/tmx'}
            data_dir = os.path.join(GENERAL_DATA_DIR, self.dataset_name)
            os.makedirs(data_dir, exist_ok=True)

            if self.tmx:
                ##http://opus.nlpl.eu/download.php?f=Europarl/v7/tmx/de-en.tmx.gz

                append_string = self.lang_code + "-" + "en" + ".tmx" + ".gz"
                url = os.path.join(self.datasets.get(self.dataset_name).get("url"), append_string)
                # url = data_url_tmx_opus + language_code + "-" + "en" + ".tmx" + ".gz"
                try:
                    raw_file = super().maybe_download_and_extract(url=url, raw_file=raw_file)
                    return raw_file
                except:
                    ##http://opus.nlpl.eu/download.php?f=Europarl/v7/tmx/en-fr.tmx.gz
                    append_string = "en" + "-" + self.lang_code + ".tmx" + ".gz"
                    url = os.path.join(self.datasets.get(self.dataset_name).get("url"), append_string)
                    raw_file = super().maybe_download_and_extract(url=url, raw_file=raw_file)
                    return raw_file


def _print_download_progress(count, block_size, total_size):
    """
    Function used for printing the download progress.
    Used as a call-back function in maybe_download_and_extract().
    """
    # Percentage completion.
    pct_complete = float(count * block_size) / total_size
    # Limit it because rounding errors may cause it to exceed 100%.
    pct_complete = min(1.0, pct_complete)
    # Status-message. Note the \r which means the line should overwrite itself.
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


########################################################################

def download_files(base_url, filename, download_dir):
    """
    Download the given file if it does not already exist in the download_dir.

    :param base_url: The internet URL without the filename.
    :param filename: The filename that will be added to the base_url.
    :param download_dir: Local directory for storing the file.
    :return: Nothing.
    """
    # Path for local file.
    save_path = os.path.join(download_dir, filename)
    # Check if the file already exists, otherwise we need to download it now.
    if not os.path.exists(save_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        print("Downloading", filename, "...")

        # Download the file from the internet.
        url = base_url + filename
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=save_path,
                                                  reporthook=_print_download_progress)

        print(" Done!")


def maybe_download_and_extract(url, download_dir, language_code="de", raw_file="{}-en.tmx"):
    """
    This function has been modified to handle tmx files.

    Download and extract the data if it doesn't already exist.
    Assumes the url is a tar-ball file.
    :param url:
        Internet URL for the tar-file to download.
        Example: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    :param download_dir:
        Directory where the downloaded file is saved.
    :return:
        Nothing.
    """

    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)  # e.g. ./data/raw/europarl/de/en-de.tmx.gz
    if not os.path.exists(file_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Download the file from the internet.
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path,
                                                  reporthook=_print_download_progress)

        print("\nFile downloaded in:", file_path)  # ./data/raw/europarl/de/en-de.tmx.gz

        print()
        print("Download finished. Extracting files.")

        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)
        elif file_path.endswith(".gz"):
            # Modified for tmx files #
            raw_file = raw_file.format(language_code)  # e.g. de-en.tmx
            download_dir = os.path.join(download_dir, raw_file)
            with gzip.open(file_path, 'rb') as gz:
                with open(download_dir, 'wb') as uncompressed:
                    shutil.copyfileobj(gz, uncompressed)
            return raw_file

        print("Done.")
    else:
        print("Data already downloaded and unpacked.")
        return filename

########################################################################
