import io
import time
import torchtext.data as data
from torchtext.data import Dataset
from torchtext.datasets import TranslationDataset

from TMX2Corpus.tmx2corpus import FileOutput
import os
from settings import DATA_DIR, DATA_DIR_RAW, DATA_DIR_PREPRO
from project.utils.data.preprocessing import MaxLenFilter, MinLenFilter, TMXConverter, get_custom_tokenizer
from project.utils.utils import convert


def read_file(path):
    pass


class Seq2SeqDataset(Dataset):
    """Defines a dataset for machine translation."""

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
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            for i, (src_line, trg_line) in enumerate(zip(src_file, trg_file)):

                if reduce >0 and i == reduce:
                    break
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    if truncate > 0:
                        src_line, trg_line = src_line.split(" "),trg_line.split(" ")
                        src_line = src_line[:truncate]
                        trg_line = trg_line[:truncate]
                        assert len(src_line) == len(trg_line)
                        src_line = ' '.join(src_line)
                        trg_line = ' '.join(trg_line)

                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))
        return examples

    @classmethod
    def splits(cls, path=None, root='.data', train=None, validation=None,
               test=None, reduce = [0,0,0], **kwargs):

        if path is None:
            path = cls.download(root)
        train_data, val_data, test_data = None, None, None
        exts = kwargs["exts"]
        reduce_samples = reduce
        fields = kwargs["fields"]
        truncate = kwargs.get("truncate", 0)
        if train:
            train_data = cls(os.path.join(path, train), exts=exts, reduce=reduce_samples[0], truncate=truncate, fields=fields)
        if validation:
            val_data = cls(os.path.join(path, validation), exts=exts, reduce=reduce_samples[1], truncate=truncate, fields=fields)
        if test:
            test_data = cls(os.path.join(path, test), exts=exts, reduce=reduce_samples[2], truncate=truncate, fields=fields)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


if __name__ == '__main__':
    #### preprocessing pipeline for tmx files
    start = time.time()
    converter = TMXConverter(output=FileOutput(path=os.path.join(DATA_DIR_PREPRO,"europarl", "de")))
    tokenizers = [get_custom_tokenizer("en", "w"), get_custom_tokenizer("de", "w")]
    converter.add_tokenizers(tokenizers)
    converter.add_filter(MaxLenFilter(30))
    converter.add_filter(MinLenFilter(5))
    converter.convert([os.path.join(DATA_DIR_RAW,"europarl", "de", "de-en.tmx")])
    print("Total time:", convert(time.time() - start))
    print(converter.output_lines)