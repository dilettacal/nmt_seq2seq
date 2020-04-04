import re
import time
import urllib

from project.utils.data import split_data, persist_txt
from project.utils.external.europarl import maybe_download_and_extract_dataset
from project.utils.external.tmx_to_text import Converter, FileOutput
from project.utils.get_tokenizer import get_custom_tokenizer
from project.utils.tokenizers import SpacyTokenizer
from project.utils.utils import convert_time_unit, Logger
from settings import DATA_DIR_RAW, DATA_DIR_PREPRO, CONFIG_PATH
import yaml
import os

from project.utils.datasets import TMXDataset

flatten = lambda l: [item for sublist in l for item in sublist]

def get_datasets(config, names):
    assert isinstance(config, list)
    datasets = []
    for i, dataset in enumerate(config):
        name = names[i]
        ds_dict = dataset[names[i]]
        genre = ds_dict["genre"]
        version = ds_dict["version"]
        url = ds_dict["url"]

        ds = TMXDataset(name=name, genre=genre, version=version, url=url)
        datasets.append(ds)

    return datasets


def preprocess_single_dataset(dataset, lang_code, parser):
    if lang_code == "en":
        raise SystemExit("English is the default language. Please provide second language!")
    if not lang_code:
        raise SystemExit("Empty language not allowed!")
        # Download the raw tmx file
    try:
        print("Trying to download the file ...")
        maybe_download_and_extract_dataset(dataset=dataset)
        # maybe_download_and_extract_europarl(language_code=lang_code, tmx=True)
    except urllib.error.HTTPError as e:
        print(e)
        raise SystemExit(
            "Please download the parallel corpus manually from: http://opus.nlpl.eu/ | Europarl > Statistics and TMX/Moses Download "
            "\nby selecting the data from the upper-right triangle (e.g. en > de])")

    path_to_raw_file = os.path.join(DATA_DIR_RAW, dataset.name, lang_code)
    MAX_LEN, MIN_LEN = 30, 2  # min_len is by defaul 2 tokens

    file_name = lang_code + "-" + "en" + ".tmx"
    COMPLETE_PATH = os.path.join(path_to_raw_file, file_name)
    print(COMPLETE_PATH)

    STORE_PATH = os.path.join(os.path.expanduser(DATA_DIR_PREPRO), dataset.name, lang_code, "splits", str(MAX_LEN))
    os.makedirs(STORE_PATH, exist_ok=True)

    start = time.time()
    output_file_path = os.path.join(DATA_DIR_PREPRO, dataset.name, lang_code)

    # Conversion tmx > text
    converter = Converter(output=FileOutput(output_file_path))
    converter.convert([COMPLETE_PATH])
    print("Converted lines:", converter.output_lines)
    print("Extraction took {} minutes to complete.".format(convert_time_unit(time.time() - start)))

    target_file = "bitext.{}".format(lang_code)
    src_lines, trg_lines = [], []

    # Read converted lines for further preprocessing
    with open(os.path.join(output_file_path, "bitext.en"), 'r', encoding="utf8") as src_file, \
            open(os.path.join(output_file_path, target_file), 'r', encoding="utf8") as target_file:
        for src_line, trg_line in zip(src_file, target_file):
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            if src_line != "" and trg_line != "":
                src_lines.append(src_line)
                trg_lines.append(trg_line)

    ### tokenize lines ####
    assert len(src_lines) == len(trg_lines), "Lines should have the same lengths."

    TOKENIZATION_MODE = "w"
    PREPRO_PHASE = True
    # Get tokenizer
    src_tokenizer, trg_tokenizer = get_custom_tokenizer("en", TOKENIZATION_MODE, prepro=PREPRO_PHASE), \
                                   get_custom_tokenizer(lang_code, TOKENIZATION_MODE, prepro=PREPRO_PHASE)

    # Creates logger to log tokenized objects
    src_logger = Logger(output_file_path, file_name="bitext.tok.en")
    trg_logger = Logger(output_file_path, file_name="bitext.tok.{}".format(lang_code))

    temp_src_toks, temp_trg_toks = [], []

    # Start the tokenisation process
    if isinstance(src_tokenizer, SpacyTokenizer):
        print("Tokenization for source sequences is performed with spaCy")
        with src_tokenizer.nlp.disable_pipes('ner'):
            for i, doc in enumerate(src_tokenizer.nlp.pipe(src_lines, batch_size=1000)):
                tok_doc = ' '.join([tok.text for tok in doc])
                temp_src_toks.append(tok_doc)
                src_logger.log(tok_doc, stdout=True if i % 100000 == 0 else False)
    else:
        print("Tokenization for source sequences is performed with FastTokenizer")
        for i, sent in enumerate(src_lines):
            tok_sent = src_tokenizer.tokenize(sent)
            tok_sent = ' '.join(tok_sent)
            temp_src_toks.append(tok_sent)
            src_logger.log(tok_sent, stdout=True if i % 100000 == 0 else False)

    if isinstance(trg_tokenizer, SpacyTokenizer):
        print("Tokenization for target sequences is performed with spaCy")
        with trg_tokenizer.nlp.disable_pipes('ner'):
            for i, doc in enumerate(trg_tokenizer.nlp.pipe(trg_lines, batch_size=1000)):
                tok_doc = ' '.join([tok.text for tok in doc])
                temp_trg_toks.append(tok_doc)
                trg_logger.log(tok_doc, stdout=True if i % 100000 == 0 else False)
    else:
        print("Tokenization for target sequences is performed with FastTokenizer")
        for i, sent in enumerate(trg_lines):
            tok_sent = trg_tokenizer.tokenize(sent)
            tok_sent = ' '.join(tok_sent)
            temp_src_toks.append(tok_sent)
            src_logger.log(tok_sent, stdout=True if i % 100000 == 0 else False)

    # Reduce lines by max_len
    filtered_src_lines, filtered_trg_lines = [], []
    print("Reducing corpus to sequences of min length {} max length: {}".format(MIN_LEN, MAX_LEN))

    filtered_src_lines, filtered_trg_lines = [], []
    for src_l, trg_l in zip(temp_src_toks, temp_trg_toks):
        ### remove possible duplicate spaces
        src_l_s = re.sub(' +', ' ', src_l)
        trg_l_s = re.sub(' +', ' ', trg_l)
        if src_l_s != "" and trg_l_s != "":
            src_l_spl, trg_l_spl = src_l_s.split(" "), trg_l_s.split(" ")
            if len(src_l_spl) <= MAX_LEN and len(trg_l_spl) <= MAX_LEN:
                if len(src_l_spl) >= MIN_LEN and len(trg_l_spl) >= MIN_LEN:
                    filtered_src_lines.append(' '.join(src_l_spl))
                    filtered_trg_lines.append(' '.join(trg_l_spl))

        assert len(filtered_src_lines) == len(filtered_trg_lines)

    src_lines, trg_lines = filtered_src_lines, filtered_trg_lines
    print("Splitting files...")
    train_data, val_data, test_data, samples_data = split_data(src_lines, trg_lines, val_ratio=parser.test_ratio)
    persist_txt(train_data, STORE_PATH, "train.tok", exts=(".en", "." + lang_code))
    persist_txt(val_data, STORE_PATH, "val.tok", exts=(".en", "." + lang_code))
    persist_txt(test_data, STORE_PATH, "test.tok", exts=(".en", "." + lang_code))
    if lang_code != "de":  # for german language sample files are versioned with the program
        print("Generating samples files...")
        persist_txt(samples_data, STORE_PATH, file_name="samples.tok", exts=(".en", "." + lang_code))

    print("Total time:", convert_time_unit(time.time() - start))


def raw_preprocess(parser):
    # configurations
    CORPUS = parser.dataset
    lang_code = parser.lang_code.lower()
    ## read dataset configs

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        datasets = config['tmx_datasets']
        tmx_datasets = get_datasets(datasets, flatten(datasets))

    available_names = [ds.name for ds in tmx_datasets]
    assert CORPUS in available_names, "Provide valid corpus name!"
    datasets = [dataset for dataset in tmx_datasets if dataset.name == CORPUS]

    for dataset in datasets:
        print("Preprocessing dataset ", dataset.name)
        preprocess_single_dataset(dataset, lang_code, parser)

