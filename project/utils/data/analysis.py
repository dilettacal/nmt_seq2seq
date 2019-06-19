import time
from collections import Counter

from project.utils.data.preprocessing import *
from project.utils.download.europarl import load_data
from project.utils.utils import Logger, convert
from settings import DATA_DIR_RAW


def count_word_frequency(data):
    word_frequency = {}
    """calculate the world frequency for each token in the corpus"""
    for text in data:
        for token in text.split():
            if token not in word_frequency:
                word_frequency[token] = 1
            else:
                word_frequency[token] += 1
    return word_frequency

def unk_counter(data, stoi):
    """Count <UNK> tokens in data"""
    unk_count = 0
    for token in data:
        if token == stoi['<UNK>']:
            unk_count += 1
    return unk_count


def sequence_length(sequence, char_level=False):
    assert isinstance(sequence, str)
    sequence = sequence.strip()
    return len(sequence.split(" ")) if not char_level else len(sequence)


def count_functional_content_words(sequences, lang):
    assert isinstance(sequences, list)
    f, c = [], []
    if lang == "en":
        from spacy.lang.en.stop_words import STOP_WORDS
        stop_words = STOP_WORDS
    elif lang == "de":
        from spacy.lang.de.stop_words import STOP_WORDS
        stop_words = STOP_WORDS
    elif lang == "fr":
        from spacy.lang.fr.stop_words import STOP_WORDS
        stop_words = STOP_WORDS
    elif lang == "it":
        from spacy.lang.it.stop_words import STOP_WORDS
        stop_words = STOP_WORDS
    else: raise("Language not supported!")

    for seq in sequences:
        s = seq.split()
        for word in s:
            if word.lower() in stop_words:
                f.append(word)
            else:
                c.append(word)

    functional = Counter(f)
    content = Counter(c)
    return functional, content


def get_max_sent_len(sentences, char_level=False):
    lengths = [(len(sent.split(" ")), i) for i, sent in enumerate(sentences)] if not char_level else [(len(sent), i) for
                                                                                                      i, sent in
                                                                                                      enumerate(
                                                                                                          sentences)]
    return max(filter(lambda x: x[0], lengths))


def get_min_sent_len(sentences, char_level=False):
    lengths = [(len(sent.split(" ")), i) for i, sent in enumerate(sentences)] if not char_level else [(len(sent), i) for
                                                                                                      i, sent in
                                                                                                      enumerate(
                                                                                                          sentences)]
    return min(filter(lambda x: x[0], lengths))


def count_words(sentences):
    # print(sentences)
    if isinstance(sentences, list):
        sentences = flatten([sent.split(" ") for sent in sentences])
    elif isinstance(sentences, str):
        sentences = sentences.split(" ")
    word_counter = Counter(sentences)
    return word_counter


def generate_csv_info_file(path, functional, content, logger, lang):


    functional_df = pd.DataFrame.from_dict(functional, orient='index').reset_index()

    functional_df.rename(columns={"index": "word", 0: "count"}).reset_index()
    mapping = {functional_df.columns[0]: "word", functional_df.columns[1]: "count"}
    functional_df = functional_df.rename(columns=mapping)
    functional_df = functional_df.sort_values(by=["count"], ascending=False)
    functional_df = functional_df.reset_index()
    functional_df.to_csv(os.path.join(path, "{}_functional_words.csv".format(lang.upper())), encoding="utf-8")

    content_df = pd.DataFrame.from_dict(content, orient='index').reset_index()
    mapping = {content_df.columns[0]: "word", content_df.columns[1]: "count"}
    content_df = content_df.rename(columns=mapping)
    content_df = content_df.sort_values(by=["count"], ascending=False)
    content_df = content_df.reset_index()
    content_df.to_csv(os.path.join(path, "{}_content_words.csv".format(lang.upper())), encoding="utf-8")

    content_hapaxes = content_df.loc[content_df['count'] == 1]
    functional_hapaxes = functional_df.loc[functional_df['count'] == 1]
    logger.log("{} - Content hapaxes: {}".format(lang.upper(), len(content_hapaxes)))
    logger.log("{} - Functional hapaxes: {}".format(lang.upper(), len(functional_hapaxes)))



def analyze_corpus(language_code, src_sentences: list, trg_senteces:list, logger=None):
    """
    A function to get some corpus statistics
    :param src_sentences: a list of sentences
    :param parse_trees: if parse tree depth for each sequence should be computed or not (this may be slow)
    :return:
    """
    ### Tokenize sentences
    preprocessed_sentences = list(preprocess_corpus(src_sents=src_sentences, trg_sents=trg_senteces, language_code=language_code))
    src_sents = [pair[0] for pair in preprocessed_sentences]
    trg_sents = [pair[1] for pair in preprocessed_sentences]

    src_word_counts = count_words(src_sents)
    trg_word_counts = count_words(trg_sents)

    ### get min and max sent length
    max_src_len, min_src_len = get_max_sent_len(src_sents, char_level=False), get_min_sent_len(src_sents,
                                                                                             char_level=False)  # (40,4), length and index
    max_trg_len, min_trg_len = get_max_sent_len(trg_sents, char_level=False), get_min_sent_len(trg_sents,
                                                                                               char_level=False)  # (40,4), length and index
    src_functional, src_content = count_functional_content_words(src_sents, lang="en")
    trg_functional, trg_content = count_functional_content_words(trg_sents, lang=language_code)

    max_src_char_seq_len, min_src_char_seq_len = get_max_sent_len(src_sents, char_level=True), get_min_sent_len(
        src_sents, char_level=True)

    max_trg_char_seq_len, min_trg_char_seq_len = get_max_sent_len(trg_sents, char_level=True), get_min_sent_len(
        trg_sents, char_level=True)

    logger.log("Source info:")

    logger.log("Total number of samples: {}".format(len(src_sentences)))
    logger.log("Total number of words: {}".format(len(src_word_counts.items())))
    logger.log("Minimum sentence length: {}".format(min_src_len))
    logger.log("Maximum sentence length: {}".format(max_src_len))
    logger.log("Minimum sentence length (char_level): {}".format(min_src_char_seq_len))
    logger.log("Maximum sentence length (char_level): {}".format(max_src_char_seq_len))
    logger.log("Number of functional words: {}".format(len(src_functional.items())))
    logger.log("Number of content words: {}".format(len(src_content.items())))

    logger.log("Target info:")

    logger.log("Total number of samples: {}".format(len(trg_senteces)))
    logger.log("Total number of words: {}".format(len(trg_word_counts.items())))
    logger.log("Minimum sentence length: {}".format(min_trg_len))
    logger.log("Maximum sentence length: {}".format(max_trg_len))
    logger.log("Minimum sentence length (char_level): {}".format(min_trg_char_seq_len))
    logger.log("Maximum sentence length (char_level): {}".format(max_trg_char_seq_len))
    logger.log("Number of functional words: {}".format(len(trg_functional.items())))
    logger.log("Number of content words: {}".format(len(trg_content.items())))

    ### dataframes for functional and content
    path = logger.path
    generate_csv_info_file(path, src_functional, src_content, logger, lang="en")
    generate_csv_info_file(path, trg_functional, trg_content, logger, lang=language_code)


if __name__ == '__main__':

    language_code = "de"

    src_data = load_data(english=True, language_code="de", tmx=True)
    trg_data = load_data(english=False, language_code="de", tmx=True)

    print(src_data[0], trg_data[0])
    print(len(src_data), len(trg_data))

    assert len(src_data) == len(trg_data)

    logger = Logger(path=get_full_path(DATA_DIR_RAW, "europarl", language_code), mode="a", file_name="Dataset.log")

    start = time.time()

    analyze_corpus(src_sentences=src_data, trg_senteces=trg_data, logger=logger, language_code=language_code)
    end = time.time()

    logger.log("Duration: {}".format(convert(end - start)))
