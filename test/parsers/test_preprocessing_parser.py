import unittest
from project.utils.preprocess import raw_preprocess
from project.utils.parsers.get_preprocess_parser import data_prepro_parser


class PreprocessParserTest(unittest.TestCase):

    def setUp(self):
        self.parser = data_prepro_parser()

    def test_valid_lang_code(self):
        parsed = self.parser.parse_args(["--lang_code", "de"])
        self.assertEqual(parsed.lang_code, "de")

    def test_empty_lang_code(self):
        with self.assertRaises(SystemExit) as sysexit:
            raw_preprocess(self.parser.parse_args(["--lang_code", ""]))
        self.assertIn('Empty language not allowed!', str(sysexit.exception))
       # self.assertRaises(ValueError, self.parser.parse_args())

    def test_en_lang_code(self):
        with self.assertRaises((SystemExit)) as sysexit:
            raw_preprocess( self.parser.parse_args(["--lang_code", "en"]))
        self.assertIn('Please provide second language!', str(sysexit.exception))

    def test_not_existing_lang_code(self):
        with self.assertRaises((SystemExit)) as sysexit:
            raw_preprocess( self.parser.parse_args(["--lang_code", "asfhah"]))
        self.assertIn("download the parallel corpus manually", str(sysexit.exception))




if __name__ == '__main__':
    unittest.main()
