import os
import unittest
import mock

from project.utils.experiment import Experiment
from translate import translate
path_to_model = os.path.expanduser("trained_model")
user_input = "Die europäische Union ist groß."

class TestTranslation(unittest.TestCase):

    def test_load_data_for_translation(self):

        with mock.patch('builtins.input', side_effect=user_input):
            results = translate(path=path_to_model, beam_size=5, predict_from_file="")

        self.assertIsInstance(results, list)


    def test_live_translation(self):
        pass

    def test_translation_from_file(self):
        pass

