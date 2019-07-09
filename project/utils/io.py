import io
import torch
import torchtext
import torchtext.data as data
from torchtext.data import Dataset
import os
from project.utils.constants import PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN

import random

from settings import SEED

random.seed(SEED)

class NMTField(torchtext.data.Field):

    def __init__(self, **kwargs):
            ### Create vocabulary object
            super(NMTField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
            super(NMTField, self).build_vocab(*args, **kwargs)
            self.sos_id = self.vocab.stoi[self.init_token]
            self.eos_id = self.vocab.stoi[self.eos_token]
            self.pad_id = self.vocab.stoi[self.pad_token]

            self.unk_id = self.vocab.stoi[self.unk_token]

    def reverse(self, batch):
            """
            Readapted from: https://github.com/pytorch/text/blob/master/torchtext/data/field.py
            Reverses the given batch back to the sentences (strings)
            """
            if self.include_lengths:
                batch = batch[0]  # if lenghts are included, batch is a tuple containing an array of all the lengths

            if not self.batch_first:
                ### batch needs to be transposed, if shape is seq_len x batch
                batch = batch.t()

            with torch.cuda.device_of(batch):
                batch = batch.tolist()

            batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]

            def trim(sent, token):
                """
                Removes from the given sentence the given token
                :param sent:
                :param token:
                :return: tokenized sentence array without the given token
                """
                sentence = []
                for word in sent:
                    if word == token:
                        break
                    sentence.append(word)
                return sentence

            batch = [trim(ex, self.vocab.itos[self.eos_id]) for ex in batch]

            def filter_special(token):
                return token not in (self.init_token, self.pad_token)

            batch = [filter(filter_special, ex) for ex in batch]

            return [' '.join(ex) for ex in batch]  ## Reverse tokenization by joining the words


class SrcField(NMTField):

    def __init__(self,sos_eos_pad_unk =[None, None, PAD_TOKEN, UNK_TOKEN], include_lengths = False, sequential=True, tokenize=None, pad_first=False,lower=False):
        self.sos_token = sos_eos_pad_unk[0]
        self.eos_token = sos_eos_pad_unk[1]
        self.pad_token = sos_eos_pad_unk[2]
        self.unk_token = sos_eos_pad_unk[3]
        super().__init__(lower=lower, pad_token=self.pad_token,
                         eos_token=self.eos_token, init_token=self.sos_token,
                         unk_token=self.unk_token, include_lengths=include_lengths,
                         sequential=sequential, tokenize=tokenize, pad_first=pad_first)


class TrgField(NMTField):

    def __init__(self, sos_eos_pad_unk =[SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN], include_lengths = False, sequential=True, tokenize=None, lower=False):
        self.sos_token = sos_eos_pad_unk[0]
        self.eos_token = sos_eos_pad_unk[1]
        self.pad_token = sos_eos_pad_unk[2]
        self.unk_token = sos_eos_pad_unk[3]
        super().__init__(lower=lower, pad_token=self.pad_token,
                         eos_token=self.eos_token, init_token=self.sos_token,
                         unk_token=self.unk_token, include_lengths=include_lengths, sequential=sequential, tokenize=tokenize)


class Seq2SeqDataset(Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(x):
        return (len(x.src), len(x.trg))

    def __init__(self, path, exts, fields, truncate=0, reduce=0):

        if not isinstance(fields[0], (tuple, list)):
           # print(fields)
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = self._generate_examples(src_path, trg_path, fields, truncate=truncate, reduce=reduce)

        super(Seq2SeqDataset, self).__init__(examples, fields)

    def _generate_examples(self, src_path, trg_path, fields, truncate, reduce):
        examples = []
        src_lines = [line.strip("\n") for line in
                     open(os.path.join(src_path), mode="r",
                          encoding="utf-8").readlines() if line]
        trg_lines = [line.strip("\n") for line in
                     open(os.path.join(trg_path), mode="r",
                          encoding="utf-8").readlines() if line]

        assert len(src_lines) == len(trg_lines)
        combined = list(zip(src_lines, trg_lines))
       # random.shuffle(combined)

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
               test=None, reduce = [0,0,0], reverse_input = False, **kwargs):

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

