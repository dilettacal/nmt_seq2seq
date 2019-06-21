import time

import torchtext
import TMX2Corpus
from TMX2Corpus.tmx2corpus import Converter, FileOutput
from TMX2Corpus.tokenizer import PyEnTokenizer
import os

from project.utils.data.mappings import ENG_CONTRACTIONS_MAP
from project.utils.data.preprocessing import WordTokenizer, expand_contraction, MaxLenFilter, MinLenFilter, EmptyFilter, \
    TMXConverter
from project.utils.utils import convert
from settings import DATA_DIR, DATA_DIR_RAW, DATA_DIR_PREPRO

if __name__ == '__main__':
    import re
    from sacremoses import MosesTokenizer
    import spacy

    nlp = spacy.load("en_core_web_sm")


    space_before_punct = r'\s([?.!"](?:\s|$))'
    before_apos = r"\s+(['])"
    after_apos = r"(['])\s+([\w])"

    text = "This text . Is to test . How it works ! Will it! Or won ' t it ? Hmm ? Is this Sara 's bag ? you can write her at this e-mail address: sara_looks@gmail.com. Or you can visit her site: https://sara-looks.de/contacts/"
    
    print("Tokenization on raw text")
    print("Text >>>> \n", text)
    
    mos_tok = MosesTokenizer("en")
    print(mos_tok.tokenize(text))
    print(list(nlp(text)))


    text = re.sub(space_before_punct, r"\1", text)
    text = re.sub(before_apos,r"\1", text)
    text = re.sub(after_apos, r"\1\2", text)
    text = text.replace("-", "")

    text = expand_contraction(text, ENG_CONTRACTIONS_MAP)

    mos_tok = MosesTokenizer("en")
    print(mos_tok.tokenize(text))
    print(list(nlp(text)))
  #  print([(tok.text, tok.pos_) for tok in nlp(text)])

    bitext = dict({"en": "I love music, and you? Are you a music lover? Why not?", "de": "Ich liebe Musik?"})
    print(bool(len(list(filter(lambda item: len(item[1].split(" ")) >= 10, bitext.items()))) ==2))

    start = time.time()
    converter = TMXConverter(output=FileOutput(path=os.path.join(DATA_DIR_PREPRO,"europarl", "de")))
    tokenizers = [WordTokenizer("en"), WordTokenizer("de")]
    converter.add_tokenizers(tokenizers)
 #   converter.add_filter(EmptyFilter())
    converter.add_filter(MaxLenFilter(30))
    converter.add_filter(MinLenFilter(5))
    converter.convert([os.path.join(DATA_DIR_RAW,"europarl", "de", "de-en.tmx")])
    print("Total time:", convert(time.time() - start))
    print(converter.output_lines)