import os
import unittest

from torchtext.data import Field, Iterator

from project.utils.utils import Logger, AverageMeter
from project.utils.datasets import Seq2SeqDataset

data_dir = os.path.join(".", "test", "test_data")

class TestIOUtils(unittest.TestCase):

    def test_load_data(self):
        src_vocab = Field(pad_token="<p>", unk_token="<u>", lower=True)
        trg_vocab = Field(init_token="<s>", eos_token="</s>",pad_token="<p>", unk_token="<u>", lower=True )
        exts = (".de", ".en")

        samples = Seq2SeqDataset.splits(root="", path=data_dir, exts=exts,
                                        train="samples", fields=(src_vocab, trg_vocab), validation="",test="")
        self.assertIsInstance(samples, tuple)
        samples = samples[0]
        self.assertIsInstance(samples, Seq2SeqDataset)
        self.assertIsNotNone(samples.examples)
        self.assertAlmostEqual(len(samples.examples), 15)
        self.assertEqual(list(samples.fields.keys()), ["src", "trg"])

        src_vocab.build_vocab(samples)
        trg_vocab.build_vocab(samples)
        self.assertIsNotNone(src_vocab.vocab.stoi)
        self.assertIsNotNone(trg_vocab.vocab.stoi)


    def test_logger(self):
        path = os.path.join(data_dir, "log.log")
        if os.path.exists(path):
            os.remove(path)
        logger = Logger(path=data_dir)
        self.assertIsNotNone(logger)
        logger.log("test_logging", stdout=False)
        logger.log("test_second_logging", stdout=False)
        with open(path, mode="r") as f:
            content = f.read().strip().split("\n")
        self.assertEqual(content[0], "test_logging")
        self.assertEqual(content[1], "test_second_logging")

    def test_save_model(self):
        path = os.path.join(data_dir, "log.log")
        if os.path.exists(path):
            os.remove(path)
        logger = Logger(path=data_dir)
        self.assertIsNotNone(logger)
        model = dict({"model": [1,2,3,4,2]})
        logger.save_model(model)
        files = os.listdir(data_dir)
        self.assertIn("model.pkl", files)
        os.remove(os.path.join(data_dir, "model.pkl"))


    def test_plot_metrics(self):
        path = os.path.join(data_dir, "log.log")
        if os.path.exists(path):
            os.remove(path)
        logger = Logger(path=data_dir)
        self.assertIsNotNone(logger)
        metric = [1,2,5,1,6,1]
        logger.plot(metric, "", "", "metric")
        files = os.listdir(data_dir)
        self.assertIn("metric.png", files)
        os.remove(os.path.join(data_dir, "metric.png"))

    def test_metric(self):
        metric = AverageMeter()
        for i in range(10):
            metric.update(i)
        self.assertEqual(metric.count, 10)
        self.assertEqual(metric.val, 9)
        self.assertEqual(metric.avg, 4.5)
        self.assertEqual(metric.sum, 45)
        metric.reset()
        self.assertEqual(metric.count, 0)
        self.assertEqual(metric.val,  0)
        self.assertEqual(metric.avg,  0)
        self.assertEqual(metric.sum, 0)