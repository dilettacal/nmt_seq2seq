"""
This script is used to run a trained model on a source file.

Load the model with torch.load, retrieve model information from the checkpoint.
Open the file and process each sentence with the method "predict_from_input".

"""
import os

import torch

from project.experiment.setup_experiment import Experiment
from settings import RESULTS_DIR

path_to_model = os.path.join(RESULTS_DIR,"/results/en_de/custom/1/uni/2019-06-25-19-31-11/custom-model.pkl")
print(path_to_model)


exp_path = "/home/dcal/Programming/python/nmt_project/nmt_thesis/nmt_seq2seq/results/en_de/custom/4/uni/2019-06-25-20-01-13"

experiment = Experiment()

experiment = torch.load(os.path.join(exp_path, "experiment.pkl"))

print(experiment)