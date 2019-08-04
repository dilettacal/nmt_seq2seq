import unittest
import spacy

from project.utils.get_tokenizer import get_custom_tokenizer
from project.utils.tokenizers import *

class TestEnvironmentTokenizers(unittest.TestCase):

    def test_factory_spacy(self):
        tokenizer = get_custom_tokenizer("de", prepro=True, mode="w")
        self.assertIsInstance(tokenizer, SpacyTokenizer)
        self.assertEqual(tokenizer.lang, "de")
        self.assertIsInstance(tokenizer.nlp, spacy.lang.de.German)
        self.assertIs(tokenizer.only_tokenize, True)

    def test_factory_spacy_en(self):
        tokenizer = get_custom_tokenizer("en", prepro=True, mode="w")
        self.assertIsInstance(tokenizer, SpacyTokenizer)
        self.assertEqual(tokenizer.lang, "en")
        self.assertIsInstance(tokenizer.nlp, spacy.lang.en.English)
        self.assertIs(tokenizer.only_tokenize, True)

    def test_factory_spacy_xx(self):
        tokenizer = get_custom_tokenizer("xx", prepro=True, mode="w")
        self.assertIsInstance(tokenizer, SpacyTokenizer)
        self.assertEqual(tokenizer.lang, "xx")
        self.assertIsInstance(tokenizer.nlp, spacy.lang.xx.MultiLanguage)
        self.assertIs(tokenizer.only_tokenize, True)

    def test_factory_split_lang(self):
        tokenizer = get_custom_tokenizer("de", prepro=False, mode="w")
        self.assertIsInstance(tokenizer, SplitTokenizer)
        self.assertEqual(tokenizer.lang, "de")
        self.assertIs(tokenizer.only_tokenize, True)

    def test_factory_split_xx(self):
        tokenizer = get_custom_tokenizer("xx", prepro=False, mode="w")
        self.assertIsInstance(tokenizer, SplitTokenizer)
        self.assertEqual(tokenizer.lang, "xx")
        self.assertIs(tokenizer.only_tokenize, True)

    def test_fale_language(self):
        tokenizer = get_custom_tokenizer("adagawe", prepro=True, mode="w")
        self.assertIsInstance(tokenizer, SpacyTokenizer)
        self.assertEqual(tokenizer.lang, "adagawe")
        self.assertIsInstance(tokenizer.nlp, spacy.lang.xx.MultiLanguage)
        self.assertIs(tokenizer.only_tokenize, True)

    def test_char_mode(self):
        tokenizer = get_custom_tokenizer("xx", prepro=False, mode="c")
        self.assertIsInstance(tokenizer, CharBasedTokenizer)

    def test_spacy_tokenizer(self):
        tokenizer = get_custom_tokenizer("de", prepro=True, mode="w")
        self.assertIsInstance(tokenizer, SpacyTokenizer)
        test_string = "das ist ein Satz"
        self.assertIsInstance(tokenizer.tokenize(test_string), list)
        self.assertIs(len(tokenizer.tokenize(test_string)), 4)

    def test_FastTokenizer(self):
        tokenizer = FastTokenizer(lang="xx")
        self.assertIsInstance(tokenizer, FastTokenizer)
        test_string = "das ist ein Satz"
        self.assertIsInstance(tokenizer.tokenize(test_string), list)
        self.assertIs(len(tokenizer.tokenize(test_string)), 4)

    def test_Chartokenizer(self):
        tokenizer = CharBasedTokenizer(lang="xx")
        self.assertIsInstance(tokenizer, CharBasedTokenizer)
        test_string = "das ist ein Satz"
        self.assertIsInstance(tokenizer.tokenize(test_string), list)
        self.assertAlmostEqual(len(tokenizer.tokenize(test_string)), 16)

    def test_SplitTokenizer(self):
        tokenizer = SplitTokenizer(lang="xx")
        self.assertIsInstance(tokenizer, SplitTokenizer)
        test_string = "das ist ein Satz"
        self.assertIsInstance(tokenizer.tokenize(test_string), list)
        self.assertAlmostEqual(len(tokenizer.tokenize(test_string)), 4)


if __name__ == '__main__':
    unittest.main()
