import os
import subprocess
import sys
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

    def save_model(self, model_dict):
        torch.save(model_dict, os.path.join(self.path, "model.pkl"))

    def pickle_obj(self, obj_dict, name):
        torch.save(obj_dict, os.path.join(self.path, str(name + ".pkl")), pickle_module=dill)

    def persist_translations(self, src, trg, preds):
        pass

    def load(self):
        if not os.path.isfile(os.path.join(self.path, self.file_name)): raise Exception("File does not exist!")
        return open(os.path.join(self.path, self.file_name)).read().split("\n")

    def plot(self, metric, title, ylabel, file, log_every=None):
        import matplotlib.pyplot as plt
        name = str(file + ".png")
        save_path = os.path.join(self.path, name)
        if isinstance(metric, dict):
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel('epoch')
            keys = list(metric.keys())
          #  print(keys)
            labels = [keys[0], keys[1]]
            values = list(metric.values())
            assert len(values) == 2 # one array for train, one for validation
            if len(values[0]) != len(values[1]):
                if(len(values[0])) > len(values[1]):
                    idx = np.arange(0, len(values[1]))*log_every
                    x1 = np.take(values[0], idx)
                    x2 = np.linspace(0, 1, len(values[1]))
                else:
                    idx = np.arange(0, len(values[0]))*log_every
                    x2 = np.take(values[1], idx)
                    x1 = np.linspace(0, 1, len(values[0]))
            else:
                x1 = np.linspace(0, 1, len(values[0]))
                x2 = np.linspace(0, 1, len(values[1]))
            #x1 = np.arange(len(values[0]))
            plt.plot(x1, metric.get(labels[0]), color="r", label = labels[0])
            plt.plot(x2, metric.get(labels[1]), color="b", label = labels[1])
            plt.legend(loc='upper right')
            plt.savefig(save_path, format="png", dpi=300)
            plt.close()
            self.log("Plot saved: {}".format(save_path))

        elif isinstance(metric, list):
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel('epoch')
            plt.plot(metric, label=ylabel, color="b")
            plt.legend(loc='upper right')
            plt.savefig(save_path,  format="png", dpi=300)
            plt.close()
            self.log("Plot saved: {}".format(save_path))

        else:raise Exception("Provide metric either as list or as dictionary.")


class Metric(object):
    def __init__(self, name, values):
        self.name = name
        self.values = values

    def get_dict(self):
        return dict({"name": self.name, "values": self.values})

class AverageMeter():
    """
    This object is used to keep track of the values for a given metric.
    Adapted version from: https://github.com/pytorch/examples/blob/master/imagenet/main.py#L354
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val is None: val = 0
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count

def load_embeddings(np_src_file, np_trg_file):
    ''' Load pretrained embeddings '''
    if os.path.isfile(np_src_file) and os.path.isfile(np_trg_file):
        pretrained_src = torch.from_numpy(np.load(np_src_file))
        pretraiend_trg = torch.from_numpy(np.load(np_trg_file))
    else:
        raise Exception('Vectors not available to load from numpy file')
    return pretrained_src, pretraiend_trg


### installing utils #####

def install_package(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])