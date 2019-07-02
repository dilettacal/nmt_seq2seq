"""
This script is used to run a trained model on a source file.

Load the model with torch.load, retrieve model information from the checkpoint.
Open the file and process each sentence with the method "predict_from_input".

TODO

"""
import os

import torch

from project.experiment.setup_experiment import Experiment
from settings import RESULTS_DIR, ROOT

path_to_model = os.path.join(RESULTS_DIR,"en_de/s/2/uni/2019-06-29-12-32-40/model.pkl")
print(path_to_model)


exp_path = os.path.join(RESULTS_DIR, "en_de/s/2/uni/2019-06-29-12-32-40/")

experiment = Experiment()

experiment = torch.load(os.path.join(exp_path, "experiment.pkl"))

train_losses = torch.load(os.path.join(exp_path, "bleus.pkl"))

print((train_losses["values"]))

#print(experiment)