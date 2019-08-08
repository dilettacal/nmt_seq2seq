import unittest

from project.utils.experiment import Experiment
from run_custom_nmt import experiment_parser


class ExperimentTest(unittest.TestCase):

    def setUp(self):
        self.parser = experiment_parser()
        print(self.parser.parse_known_args())

    def test_valid_experiment(self):
       # self.parser.parse_args(["--lr", float(0.002)])
        self.parser.parse_args(["--hs", 500])
        self.parser.parse_args(["--emb", 500])
        self.parser.parse_args(["--train", 500])
        self.parser.parse_args(["--test", 500])
        self.parser.parse_args(["--val", 500])

        experiment = Experiment(self.parser)
        self.assertEqual(experiment.train_samples, 500)
        self.assertEqual(experiment.val_samples, 500)
        self.assertEqual(experiment.test_samples, 500)
        self.assertEqual(experiment.emb_size, 500)
        self.assertEqual(experiment.hid_dim, 500)
        self.assertEqual(experiment.lr,  0.002)

if __name__ == '__main__':
    unittest.main()
