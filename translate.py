"""
This script is used to run a trained model on a source file.

Load the model with torch.load, retrieve model information from the checkpoint.
Open the file and process each sentence with the method "predict_from_input".

TODO

"""
import os

import torch

from project.model.models import get_nmt_model
from run_custom_nmt import experiment_parser
from project.utils.constants import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from project.utils.experiment import Experiment
from project.utils.training import predict_from_input
from project.utils.utils import load_embeddings, Logger
from project.utils.vocabulary import get_vocabularies_iterators
from settings import RESULTS_DIR, ROOT


path_to_exp = os.path.join(RESULTS_DIR, "baseline", "de_en/s/2/uni/2019-07-07-16-27-12/")

path_to_model = os.path.join(path_to_exp, "model.pkl")
print(path_to_model)


#experiment = Experiment(experiment_parser())

experiment = torch.load(os.path.join(path_to_exp, "experiment.pkl"))
print(type(experiment))
print(experiment)

experiment = Experiment(experiment["args"])
print(type(experiment))

train_losses = torch.load(os.path.join(path_to_exp, "train_losses.pkl"))

print((train_losses["values"]))

"""
SRC, TRG, train_iter, val_iter, test_iter, train_data, val_data, test_data, samples, samples_iter = \
        get_vocabularies_iterators(experiment, None)

logger = Logger(path_to_exp, file_name="test.log")
logger.pickle_obj(SRC, "src")
logger.pickle_obj(TRG, "trg")


"""



SRC_loaded = torch.load(os.path.join(path_to_exp, "src.pkl"))
TRG_loaded = torch.load(os.path.join(path_to_exp, "trg.pkl"))


tokens_bos_eos_pad_unk = [TRG_loaded.vocab.stoi[SOS_TOKEN], TRG_loaded.vocab.stoi[EOS_TOKEN], TRG_loaded.vocab.stoi[PAD_TOKEN], TRG_loaded.vocab.stoi[UNK_TOKEN]]


print(len(SRC_loaded.vocab))

print(len(TRG_loaded.vocab))


Sent = "Hallo, der Mann will arbeiten in der EU"

tokenized = SRC_loaded.tokenize(Sent)
print(tokenized)
idx = [SRC_loaded.vocab.stoi[word] if word in SRC_loaded.vocab.stoi else SRC_loaded.vocab.stoi['<unk>'] for word in tokenized]

print(idx)


### loading model

experiment.src_vocab_size = len(SRC_loaded.vocab)
experiment.trg_vocab_size = len(TRG_loaded.vocab)
model = get_nmt_model(experiment, tokens_bos_eos_pad_unk)

model.load_state_dict(torch.load(os.path.join(path_to_exp, "model.pkl")))
model = model.to(experiment.get_device())

print(experiment.__dict__)
logger = Logger(path_to_exp, "live_transl.log")
predict_from_input(input_sentence=Sent, SRC=SRC_loaded, TRG=TRG_loaded, model=model, device=experiment.get_device(), logger=logger)



