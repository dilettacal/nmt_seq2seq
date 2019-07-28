import argparse
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


def convert_time_unit(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def epoch_duration(start_time, end_time):
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
        with open(os.path.join(self.path, self.file_name), "a", encoding="utf8") as f:
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

    def plot(self, metric, title, ylabel, file):
        print("Pltting with new plotting style")
        try:
            import matplotlib.pyplot as plt
        except ImportError or ModuleNotFoundError:
            print("Module matplotlib not found. Please install matplotlib!")
            return
        name = str(file + ".png")
        save_path = os.path.join(self.path, name)
        if isinstance(metric, dict):
            ### this plots 2 metrics
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel('epoch')
            keys = list(metric.keys())
            labels = [keys[0], keys[1]]
            values = list(metric.values())
            assert len(values) == 2
            assert len(values[0]) == len(values[1])
            fig = plt.figure()
            ax = plt.subplot(111)
            x = np.arange(len(values[0]))
            ax.plot(x, metric.get(labels[0]), color="r", label=labels[0])
            ax.plot(x, metric.get(labels[1]), color="b", label=labels[1])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                      ncol=2, fancybox=True, shadow=True)
            plt.savefig(save_path, format="png", dpi=500)
            plt.close()
            self.log("Plot saved: {}".format(save_path))
        elif isinstance(metric, list):
            ### this plots one metric
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel('epoch')
            plt.plot(metric, label=ylabel, color="b")
            plt.legend(loc='upper right')
            plt.savefig(save_path,  format="png", dpi=500)
            plt.close()
            self.log("Plot saved: {}".format(save_path))

        else:raise Exception("Provide metric either as list or as dictionary containing 2 metrics.")


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


def str2bool(v):
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2float(s):
    try:
        return float(s)
    except ValueError:
        return None


def str2array(s):
    if s:
        s = s.split(" ")
    return s