import os
import re
import subprocess
import tempfile
import time

import dill
import torch
import numpy as np

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)


def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Logger():
    '''Prints to a log file and to standard output'''
    def __init__(self, path, file_name="log.log"):
        if os.path.exists(path):
            self.path = path
            self.file_name = file_name
        else:
            raise Exception('path does not exist')

    def log(self, info, stdout=True):
        with open(os.path.join(self.path, self.file_name), "a") as f:
            print(info, file=f)
        if stdout:
            print(info)

    def save_model(self, model_dict, type="sutskever"):
        model_name = "{}-model.pkl".format(type)
        self.log(">>>> Path to model: {}".format(os.path.join(self.path, model_name)))

        torch.save(model_dict, os.path.join(self.path, model_name))

    def save(self, obj_dict):
        torch.save(obj_dict, os.path.join(self.path, "experiment.pkl"), pickle_module=dill)

    def load(self):
        if not os.path.isfile(os.path.join(self.path, self.file_name)): raise Exception("File does not exist!")
        return open(os.path.join(self.path, self.file_name)).read().split("\n")


class Plotter():
    def __init__(self, path, file_name):
        if os.path.exists(path):
            self.path = path
            self.file_name = file_name
        else:
            raise Exception('path does not exist')

    def plot(self, plt_object):
        pass


class AverageMeter():
    '''Computes and stores the average and current value.
       Taken from the PyTorch ImageNet tutorial'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count