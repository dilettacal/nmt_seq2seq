"""
This python file includes some special filter for a better preprocessing.
- umlaut_dictionary maps chars with diaeresis to the equivalent form without diaeresis
- eng_prefixes_dictionary maps some English contractions to the equivalent complete form

Other filters are used to filter out certain sentences and reduce the corpus size:
- eng_prefixes reduces the dataset to the sentences beginning with those prefixes
-
"""
import string

split_chars = lambda char: list(char.strip().split(" "))
merge_chars = lambda char: char.strip().replace(" ", "|")
group_chars = lambda char: char.strip().replace(" ", "")


# Mapping for German language
UMLAUT_MAP = {u'Ä': 'Ae',
                     u'Ö': 'Oe',
                     u'Ü': 'Ue',
                     u'ä': 'ae',
                     u'ö': 'oe',
                     u'ü': 'ue',
                     u"ß": 'ss'
              }

GER_CONTRACTIONS_MAP = {
    "auf's": "auf das",
    "find's": "finde es",
    "für's" : "für das",
    "gab's": "gab es",
    "geht's": "geht es",
    "gibt's": "gibt es",
    "hab'" : "habe",
    "hab's" : "habe es",
    "hat's" : "hat es",
    "ich's"  : "ich es",
    "ist's" :"ist es",
    "kann's" :"kann es",
    "macht's" :"macht es",
    "ob's": "ob es",
    "sag's" : "sage es",
    "schaut's" : "schaut es",
    "sie's" : "sie es",
    "sieht's": "sieht es",
    "sind's"  : "sind es",
    "spielt's": "spielt es",
    "tut's"  :"tut es",
    "war's"  : "war es",
    "weil's" :"weil es",
    "wenn's": "wenn es",
    "wie's" : "wie es",
    "wir's" : "wir es",
    "wird's" :"wird es",
    "wär's"  :"wäre es",
    "'nem": "einem",
    "'nen": "einen",
    "'ner": "einer",
    "aufm": "auf dem",
    "aufn": "auf den",
    "aufs": "auf das",
    "ausm": "aus dem",
    "drauf": "darauf",
    "drum": "darum",
    "fürn": "für einen",
    "fürs": "für es",
    "gibts": "gibt es",
    "haste": "hast du",
    "in's": "in das",
    "ins": "in das",
    "übers": "über das",
    "untern": "unter den",
    "unterm": "unter dem",
    "vorn" : "vorne",
    "vors" :"vor das",
}
# Contraction map for english - can be extended
# Map mostly taken from: https://github.com/kootenpv/contractions/blob/master/contractions/__init__.py
ENG_CONTRACTIONS_MAP = {
    u"'s":"",
    u"ain't": "are not",
    u"aren't": "are not",
    u"can't": "cannot",
    u"can't've": "cannot have",
    u"'cause": "because",
    u"could've": "could have",
    u"couldn't": "could not",
    u"couldn't've": "could not have",
    u"didn't": "did not",
    u"doesn't": "does not",
    u"don't": "do not",
    u"hadn't": "had not",
    u"hadn't've": "had not have",
    u"hasn't": "has not",
    u"haven't": "have not",
    u"he'd": "he would",
    u"he'd've": "he would have",
    u"he'll": "he will",
    u"he'll've": "he will have",
    u"he's": "he is",
    u"how'd": "how did",
    u"how'd'y": "how do you",
    u"how'll": "how will",
    u"how's": "how is",
    u"i'd": "i would",
    u"i'd've": "i would have",
    u"i'll": "i will",
    u"i'll've": "i will have",
    u"i'm": "i am",
    u"i've": "i have",
    u"isn't": "is not",
    u"it'd": "it would",
    u"it'd've": "it would have",
    u"it'll": "it will",
    u"it'll've": "it will have",
    u"it's": "it is",
    u"let's": "let us",
    u"mayn't": "may not",
    u"might've": "might have",
    u"mightn't": "might not",
    u"mightn't've": "might not have",
    u"must've": "must have",
    u"mustn't": "must not",
    u"mustn't've": "must not have",
    u"needn't": "need not",
    u"needn't've": "need not have",
    u"o'clock": "of the clock",
    u"she'd": "she would",
    u"she'd've": "she would have",
    u"she'll": "she will",
    u"she'll've": "she will have",
    u"she's": "she is",
    u"should've": "should have",
    u"shouldn't": "should not",
    u"shouldn't've": "should not have",
    u"so've": "so have",
    u"so's": "so is",
    u"that'd": "that would",
    u"that'd've": "that would have",
    u"that's": "that is",
    u"there'd": "there would",
    u"there'd've": "there would have",
    u"there's": "there is",
    u"they'd": "they would",
    u"they'd've": "they would have",
    u"they'll": "they will",
    u"they'll've": "they will have",
    u"they're": "they are",
    u"they've": "they have",
    u"to've": "to have",
    u"wasn't": "was not",
    u"we'd": "we would",
    u"we'd've": "we would have",
    u"we'll": "we will",
    u"we'll've": "we will have",
    u"we're": "we are",
    u"we've": "we have",
    u"weren't": "were not",
    u"what'll": "what will",
    u"what'll've": "what will have",
    u"what're": "what are",
    u"what's": "what is",
    u"what've": "what have",
    u"when's": "when is",
    u"when've": "when have",
    u"where'd": "where did",
    u"where's": "where is",
    u"where've": "where have",
    u"who'll": "who will",
    u"who'll've": "who will have",
    u"who's": "who is",
    u"who've": "who have",
    u"why's": "why is",
    u"why've": "why have",
    u"will've": "will have",
    u"won't": "will not",
    u"won't've": "will not have",
    u"would've": "would have",
    u"wouldn't": "would not",
    u"wouldn't've": "would not have",
    u"y'all're": "you all are",
    u"y'all've": "you all have",
    u"you'd": "you would",
    u"you'd've": "you would have",
    u"you'll": "you will",
    u"you'll've": "you shall have",
    u"you're": "you are",
    u"you've": "you have",
    u"doin'": "doing",
    u"goin'": "going",
    u"nothin'": "nothing",
    u"somethin'": "something",
    u"i'm ": "i am",
    u"n't": "not",
    u"'ve": "have",
    u"'ll": "will",
    u"'re": "are"
}

### See spacy

_units = (
    "h m/s km/h kmh km km² km³ m m² m³ dm dm² dm³ cm cm² cm³ mm mm² mm³ ha µm nm yd in ft"
    "kg g mg µg t lb oz mph hPa Pa mbar mb MB kb KB gb GB tb"
    "TB T G M K % км км² км³ м м² м³ дм дм² дм³ см см² см³ мм мм² мм³ нм "
)

_quotes = r'\' " ” “ ` ‘ ´ ’ ‚ , „ » « 「 」 『 』 （ ） 〔 〕 【 】 《 》 〈 〉'


_hyphens = "- – — -- --- —— ~"

_currency = r"\$ £ € ¥ ฿ US\$ C\$ A\$ ₽ ﷼ ₴ EUR"

_punct = (
    r"… …… , : ; \! \? ¿ ؟ ¡ \( \) \[ \] \{ \} < > _ # \* & 。 ？ ！ ， 、 ； ： ～ · । ، ؛ ٪"
)


UNITS = merge_chars(_units)
CURRENCY = merge_chars(_currency)
PUNCT = merge_chars(_punct)
HYPHENS = merge_chars(_hyphens)
QUOTES = merge_chars(_quotes)


LIST_UNITS = split_chars(_units)

LIST_CURRENCY = split_chars(_currency)

LIST_QUOTES = split_chars(_quotes)

TEMP_LIST_PUNCT = split_chars(_punct)
LIST_PUNCT = split_chars(_punct)

LIST_HYPHENS = split_chars(_hyphens)
LIST_ELLIPSES = [r"\.\.+", "…"]

CONCAT_QUOTES = group_chars(_quotes)
CONCAT_UNITS = group_chars(_units)
CONCAT_PUNCT = group_chars(_punct)
CONCAT_HYPHENS = group_chars(_hyphens)
CONCAT_CURRENCY = group_chars(_currency)

ALL_PUNCT = list(set(LIST_PUNCT).union(list(string.punctuation)))
MERGED_ALL_PUNCT = ''.join(ALL_PUNCT)
