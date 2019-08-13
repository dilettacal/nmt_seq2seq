import os
from torchtext import data as data
from torchtext.data import Dataset
import random
from settings import SEED
random.seed(SEED)

class Seq2SeqDataset(Dataset):
    """
    Defines a dataset for machine translation.
    Part of this code is taken from the original source code TranslationDatset:
    See: https://github.com/pytorch/text/blob/master/torchtext/datasets/translation.py#L10
    """

    @staticmethod
    def sort_key(x):
        return (len(x.src), len(x.trg))

    def __init__(self, path, exts, fields, truncate=0, reduce=0):

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = self._generate_examples(src_path, trg_path, fields, truncate=truncate, reduce=reduce)
        super(Seq2SeqDataset, self).__init__(examples, fields)

    def _generate_examples(self, src_path, trg_path, fields, truncate, reduce):
        examples = []

        src_exist = os.path.isfile(os.path.join(src_path))
        trg_exist = os.path.isfile(os.path.join(trg_path))
        if not src_exist or not trg_exist:
            return None

        print("Preprocessing files: {}, {}".format(src_path, trg_path))
        src_lines = [line.strip("\n") for line in
                         open(os.path.join(src_path), mode="r",
                              encoding="utf-8").readlines() if line]
        trg_lines = [line.strip("\n") for line in
                     open(os.path.join(trg_path), mode="r",
                          encoding="utf-8").readlines() if line]

        assert len(src_lines) == len(trg_lines)
        combined = list(zip(src_lines, trg_lines))

        for i, (src_line, trg_line) in enumerate(combined):
            src_line, trg_line = src_line.strip(), trg_line.strip()
            if src_line != '' and trg_line != '':
                if truncate > 0:
                    src_line, trg_line = src_line.split(" "), trg_line.split(" ")
                    src_line = src_line[:truncate]
                    trg_line = trg_line[:truncate]
                    assert (len(src_line) <= truncate) and (len(trg_line) <= truncate)
                    src_line = ' '.join(src_line)
                    trg_line = ' '.join(trg_line)

                examples.append(data.Example.fromlist(
                    [src_line, trg_line], fields))

            if reduce > 0 and i == reduce:
                break

        return examples

    @classmethod
    def splits(cls, path=None, root='', train=None, validation=None,
               test=None, reduce = [0,0,0], **kwargs):

        exts = kwargs["exts"]
        reduce_samples = reduce
        fields = kwargs["fields"]
        truncate = kwargs.get("truncate", 0)
        if train or train != "":
            train_data = cls(os.path.join(path, train), exts=exts, reduce=reduce_samples[0], truncate=truncate, fields=fields)
        else: train_data = None

        if validation or validation != "":
            val_data = cls(os.path.join(path, validation), exts=exts, reduce=reduce_samples[1], truncate=truncate, fields=fields)
        else: val_data = None

        if test or test != "":
            test_data = cls(os.path.join(path, test), exts=exts, reduce=reduce_samples[2], truncate=truncate, fields=fields)
        else: test_data = None
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


