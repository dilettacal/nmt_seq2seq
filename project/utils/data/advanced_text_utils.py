import re
import string
from itertools import groupby

import numpy as np
from sacremoses import MosesPunctNormalizer

from project.utils.data.mappings import ENG_CONTRACTIONS_MAP
from project.utils.data.preprocessing import expand_contraction


POS = ["PROPN", "NUM", "SYM"]
NER = ["ORG"]

def advanced_preprocess_sentence(sent, **kwargs):
    nlp = kwargs.get("nlp")
    replace_ents = kwargs.get("replace_entities", True)
    replace_special = kwargs.get("replace_special", True)
    remove_stopwords = kwargs.get("remove_stopwords", False)
    lang = kwargs.get("lang", "en")

    text = sent
    tokenizer = MosesPunctNormalizer(lang=lang)
    text = tokenizer.normalize(text)  # normalize wrong punctuation
    text = expand_contraction(text, ENG_CONTRACTIONS_MAP)

    ## Remove hyphens
    text = text.replace('-', '')

    text = expand_contraction(text, ENG_CONTRACTIONS_MAP)

    if replace_ents:
        text = replace_entities(text, nlp)

    if replace_special:
        text = replace_special_tokens(text, nlp)

    if remove_stopwords:
        text = nlp(text)
        text = [str(token.orth_) for token in text
                if not token.is_stop and not token.is_punct]
        text = ' '.join(text)
    else:
        text = nlp(text)
        text = [str(token.orth_) for token in text if not token.is_punct and token.text not in string.punctuation]
        text = ' '.join(text)

    text = final_refine_text(text)
    return text


def final_refine_text(text):
    ### remove doubled spaces
    text = text.strip()
    text = re.sub(' +', ' ', text)

    ### lowercase all but UPPERCASED words
    words = [word for word in text.split(" ") if word.isupper()]
    text = [word if word in words else word.lower() for word in text.split(" ")]
    # text = [word if len(word) > 1 and word.islower() else "" for word in text]

    ### Remove duplicates (Mr. Brown --> PROPN PROPN)
    text = [i[0] for i in groupby(text)]
    text = " ".join(text)
    return text


def replace_special_tokens(text, nlp):
    doc = nlp(text)
    text_to_pos = [(token.text, token.pos_) for token in doc if not token.text.isupper()]
    for pos in text_to_pos:
        if str(pos[1]) in POS:
            replacee = str(pos[0])
            replacer = str(pos[1]) + " "
        try:
            text = text.replace(replacee, replacer)
        except:
            pass
    return text


def convert_text_to_pos(text, nlp):
    doc = nlp(text)
    text_to_pos = [(token.text, token.pos_) for token in doc if not token.text.isupper()]
    return text_to_pos


def replace_entities(text, nlp):
    doc = nlp(text)
    text_ents = [(str(ent), str(ent.label_)) for ent in doc.ents]
    # Replace entities
    for ent in text_ents:
        replacee = str(ent[0])
        replacer = str(ent[1])
        try:
            text = text.replace(replacee, replacer)
        except:
            pass
    return text


def tree_height(root):
    """
    Find the maximum depth (height) of the dependency parse of a spacy sentence by starting with its root
    Code adapted from https://stackoverflow.com/questions/35920826/how-to-find-height-for-non-binary-tree
    :param root: spacy.tokens.token.Token
    :return: int, maximum height of sentence's dependency parse tree
    """
    if not list(root.children):
        return 1
    else:
        # print("Children", list(root.children))
        return 1 + max(tree_height(x) for x in root.children)


def get_average_heights(paragraph, nlp):
    """
    Computes average height of parse trees for each sentence in paragraph.
    :param paragraph: spacy doc object or str
    :return: float
    """
    if type(paragraph) == str:
        doc = nlp(paragraph)
    elif isinstance(paragraph, list):
        paragraph = '. '.join(paragraph)
        doc = nlp(paragraph)
    else:
        doc = paragraph
    roots = [sent.root for sent in doc.sents]
    # spacy.displacy.serve(doc, style='dep')
    return np.mean([tree_height(root) for root in roots])


def get_seq_tree_heigth(sequence, nlp):
    # print(sequence)
    if type(sequence) == str:
        doc = nlp(sequence)
    else:
        doc = sequence
    roots = [sent.root for sent in doc.sents]
    heights = [tree_height(root) for root in roots]
    num_roots = len(roots)
    return heights, num_roots


def get_dep_tree_heights(sequences, nlp):
    stats = [get_seq_tree_heigth(seq, nlp) for seq in sequences]
    # print(stats[1][0][0])
    max_value = np.max(list(map(lambda x: x[0], stats)))
    min_value = np.min(list(map(lambda x: x[0], stats)))
    mean_value = np.mean(list(map(lambda x: x[0], stats)))
    min_idx = [i for i in range(len(sequences)) if stats[i][0][0] == min_value]
    max_idx = [i for i in range(len(sequences)) if stats[i][0][0] == max_value]

    return {"max": max_value, "min": min_value, "mean": mean_value, "min_sent": min_idx[0], "max_sent": max_idx[0]}