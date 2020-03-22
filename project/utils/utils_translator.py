import os

import torch

from project.utils.constants import UNK_TOKEN, SOS_TOKEN, EOS_TOKEN


class Translator(object):
    def __init__(self, model, SRC, TRG, logger, src_tokenizer, device="cuda", beam_size=5, max_len=30):
        """
        :param model: the trained model
        :param SRC: the src vocabulary
        :param TRG: the target vocabulary
        :param logger: the translation logger
        :param src_tokenizer: the source tokenizer
        :param device: the device
        :param beam_size:
        :param max_len: unroll steps during prediction
        """
        self.model = model
        self.src_vocab = SRC
        self.trg_vocab = TRG
        self.logger = logger
        self.device = device
        self.beam_size = beam_size
        self.max_len = max_len
        self.src_tokenizer = src_tokenizer

    def predict_sentence(self, sentence, stdout=False):
        sentence = self.src_tokenizer.tokenize(sentence.lower())
        #### Changed from original ###
        sent_indices = [self.src_vocab.vocab.stoi[word] if word in self.src_vocab.vocab.stoi
                        else self.src_vocab.vocab.stoi[UNK_TOKEN] for word in
                        sentence]
        sent = torch.LongTensor([sent_indices])
        sent = sent.to(self.device)
        sent = sent.view(-1, 1)
        self.logger.log('SRC  >>> ' + ' '.join([self.src_vocab.vocab.itos[index] for index in sent_indices]),
                        stdout=stdout)
        pred = self.model.predict(sent, beam_size=self.beam_size, max_len=self.max_len)
        pred = [index for index in pred if index not in [self.trg_vocab.vocab.stoi[SOS_TOKEN],
                                                         self.trg_vocab.vocab.stoi[EOS_TOKEN]]]
        out = ' '.join(self.trg_vocab.vocab.itos[idx] for idx in pred)
        self.logger.log('PRED >>> ' + out, stdout=True)
        return out

    def predict_from_text(self, path_to_file):
        path_to_file = os.path.expanduser(path_to_file)
        self.logger.log("Predictions from file: {}".format(path_to_file))
        self.logger.log("-" * 100, stdout=True)
        # read file
        with open(path_to_file, encoding="utf-8", mode="r") as f:
            samples = f.readlines()
        samples = [x.strip().lower() for x in samples if x]
        for sample in samples:
            out = self.predict_sentence(sample, stdout=True)
            self.logger.log("-" * 100, stdout=True)

    def set_beam_size(self, new_size):
        self.beam_size = new_size

    def get_beam_size(self):
        return self.beam_size
