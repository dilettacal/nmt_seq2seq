import unittest

from translate import translate
from project.utils.parsers.get_translation_parser import translation_parser


class TranslateParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = translation_parser()

    def test_valid_path(self):
        self.assertIs(self.parser.parse_args(["--path", "results"]).path, "results")

    def test_file(self):
        self.assertIs(self.parser.parse_args(["--file", "translation.txt"]).file, "translation.txt")

    def empty_path(self):
        self.assertFalse(translate(""))