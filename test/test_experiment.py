import unittest

from project.utils.experiment import Experiment
from train_model import experiment_parser


class ExperimentTest(unittest.TestCase):

    def setUp(self):
        self.parser = experiment_parser()
        print(self.parser.parse_known_args())

    def test_default(self):
        experiment = Experiment(self.parser)
        parsed = experiment.args
        # instantiation
        self.assertEqual(parsed.train, experiment.train_samples)
        self.assertEqual(parsed.val, experiment.val_samples)
        self.assertEqual(parsed.test, experiment.test_samples)
        self.assertEqual(parsed.emb, experiment.emb_size)
        self.assertEqual(parsed.hs, experiment.hid_dim)
        self.assertEqual(parsed.lr, experiment.lr)
        self.assertEqual(parsed.max_len, experiment.truncate)
        self.assertEqual(experiment.model_type, "none")
        experiment.model_type = "custom"
        self.assertEqual(experiment.model_type, "custom")
        self.assertEqual(parsed.lang_code, "de")
        self.assertEqual(experiment.lang_code, "de")
        self.assertEqual(parsed.reverse, experiment.reverse_lang_comb)
        if experiment.lang_code == "de" and experiment.reverse_lang_comb:
            self.assertEqual(experiment.get_src_lang(), "de")
            self.assertEqual(experiment.get_trg_lang(), "en")
        else:
            self.assertEqual(experiment.get_src_lang(), "en")
            self.assertEqual(experiment.get_trg_lang(), "de")
        if experiment.bi:
            self.assertEqual(parsed.reverse_input, False)
            self.assertEqual(parsed.bi, True)
            self.assertEqual(experiment.reverse_input, False)
        else:
            self.assertEqual(parsed.reverse_input, True)
            self.assertEqual(parsed.bi, False)
            self.assertEqual(experiment.reverse_input, True)

