import os

import project
from project.utils.preprocessing import FastTokenizer, CharBasedTokenizer, SplitTokenizer, get_custom_tokenizer
from project.utils.preprocessing import SpacyTokenizer
import spacy
from settings import DATA_DIR_PREPRO
import nltk

de_nlp = spacy.load("de")
en_nlp = spacy.load("en")

frase1 = "This is John's bag, he's been very late."
frase2 = "You've been very late."
frase3 = "A pizza cost at least 12.50€ in Berlin."
frase4 = "Mr. Müller , you were unable to attend the Conference of Presidents on 28/05/2019 ."
frase5 = "I have thus proposed that the frost rating be lowered to -40ºC."
frase6 = "visit www.mysite.com/things or send an email to my.supermail@me.com"
frase7 = "The commission approved the report A-30051"
frasi = [frase1, frase2, frase3, frase4, frase5, frase6, frase7]


fast_tok = FastTokenizer("en")

en_sp = SpacyTokenizer("en", en_nlp)
de_sp = SpacyTokenizer("de", de_nlp)

for fr in frasi:
    print("Spacy:")
    print(', '.join(en_sp.tokenize(fr)))
    print([(ent.text, ent.label_) for ent in en_nlp(fr).ents])
    print([(tok.text, tok.pos_) for tok in en_nlp(fr)])
    print("Fast")
    print(', '.join(fast_tok.tokenize(fr)))
    print("NLTK")
    print(', '.join(nltk.word_tokenize(fr)))
