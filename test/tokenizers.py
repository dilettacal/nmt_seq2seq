import os

import project
from project.utils.preprocessing import FastTokenizer, CharBasedTokenizer, SplitTokenizer, get_custom_tokenizer
from project.utils.preprocessing import SpacyTokenizer
import spacy

from settings import DATA_DIR_PREPRO

output_file_path = os.path.join(DATA_DIR_PREPRO, "europarl", "de")
src_lines, trg_lines = [], []

with open(os.path.join(".", "test_tok.en"), 'r') as src_file, \
        open(os.path.join(".", "test_tok.de"), 'r') as target_file:
    for src_line, trg_line in zip(src_file, target_file):
        src_line = src_line.strip()
        trg_line = trg_line.strip()
        if src_line != "" and trg_line != "":
            src_lines.append(src_line)
            trg_lines.append(trg_line)

src_split_tok, trg_split_tok = get_custom_tokenizer("en", spacy_pretok=True), get_custom_tokenizer("de", spacy_pretok=True)
assert isinstance(src_split_tok, project.utils.preprocessing.SplitTokenizer)
assert isinstance(trg_split_tok, project.utils.preprocessing.SplitTokenizer)

for src_l, trg_l in zip(src_lines, trg_lines):
    print("Source sequence:")
    print(src_l)
    print(src_split_tok.tokenize(src_l))
    print("Target sequence:")
    print(trg_l)
    print(trg_split_tok.tokenize(trg_l))

