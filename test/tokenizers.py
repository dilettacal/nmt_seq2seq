from project.utils.preprocessing import FastTokenizer, CharBasedTokenizer, SplitTokenizer
from project.utils.preprocessing import SpacyTokenizer
from project.utils.tmx2corpus.tokenizer import PyEnTokenizer

import spacy

en_nlp = spacy.load("en")

frase = "This is my e-mail address: diletta.cal@gmail.com and it's my web site: http://dile.com. This is Robert's email: robby_m@gmail.com. This us the University URL: www.uni.de/vorlesungsverzeichnis/sose2019"
frase2 = "Diese Abstimmung ist meiner Erinnerung nach so ausgegangen: 422 gegen 180 Stimmen bei einigen wenigen Enthaltungen."
tok1 = FastTokenizer("en")
tok2 = SpacyTokenizer("en", en_nlp)
tok3 = PyEnTokenizer()

tok4 = CharBasedTokenizer("en")

print("Fast tokenizer:")
print(tok1.tokenize(frase))
print("Spacy:")
print(tok2.tokenize(frase))

print(tok4.tokenize(frase))

print("Default:")
print(tok3.tokenize(frase).split(" "))


tok5 = SplitTokenizer("en")
spacy_tok = tok2.tokenize(frase)
print(tok5.tokenize(' '.join(spacy_tok)))
print(tok5.tokenize(frase))