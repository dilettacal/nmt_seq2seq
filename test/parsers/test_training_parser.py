import unittest


import unittest

from project.utils.parsers.get_train_model_parser import experiment_parser
from translate import translate
from project.utils.parsers.get_translation_parser import translation_parser


class TrainModelParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = experiment_parser()

    def test_default_params(self):
        parsed = self.parser.parse_args()
        self.assertEqual(parsed.lr, 0.0002)
        self.assertEqual(parsed.hs, 300)
        self.assertEqual(parsed.reverse_input, False)
        self.assertEqual(parsed.bi, True)
        self.assertEqual(parsed.max_len, 30)
        self.assertEqual(parsed.epochs, 80)
        self.assertEqual(parsed.data_dir, None)
        self.assertEqual(parsed.emb, 300)
        self.assertEqual(parsed.rnn, "lstm")
        self.assertEqual(parsed.num_layers, 2)
        self.assertEqual(parsed.cuda, True)
        self.assertEqual(parsed.beam, 5)
        self.assertEqual(parsed.reverse, True)
        self.assertEqual(parsed.lang_code, "de")



if __name__ == '__main__':
    unittest.main()
