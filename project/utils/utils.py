import argparse
import os
import time
import configparser

import dill
import torch
import numpy as np

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)


class Logger():
    """
    The Logger objects logs information, pickles experiment objects and the model
    """

    def __init__(self, path, file_name="log.log"):
        if os.path.exists(path):
            self.path = path
            self.file_name = file_name
        else:
            raise Exception('path does not exist')

    def log(self, info, stdout=True):
        """
        Logs the given info to the file
        :param info: The info to log
        :param stdout: True if info should be displayed
        """
        with open(os.path.join(self.path, self.file_name), "a", encoding="utf8") as f:
            print(info, file=f)
        if stdout:
            print(info)

    def save_model(self, model_dict):
        """
        Saves the model at the path
        :param model_dict: the model dictionary
        """
        torch.save(model_dict, os.path.join(self.path, "model.pkl"))

    def pickle_obj(self, obj_dict, name):
        """
        Saves the given object dictionary with the given name
        :param obj_dict: object dictionary
        :param name: file name without format, e.g. "src_vocab"
        """
        torch.save(obj_dict, os.path.join(self.path, str(name + ".pkl")), pickle_module=dill)


    def plot(self, metric, title, ylabel, file):
        """
        Plots metrics as png images
        :param metric: provided as list or dictionary of 2 metrics
        :param title: the plot title
        :param ylabel: the y label
        :param file: file name
        """
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
            fig.savefig(save_path, format="png", dpi=500)
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
        """
        Instantiates a metric object with the given name and the given values
        :param name: name as string
        :param values: values from training
        """
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


class DatasetConfigParser(object):
    # config/datasets.ini
    def __init__(self, config_path):
        self.config_path = config_path
        self.parser = configparser.ConfigParser()
        self.parser.read(config_path)
        self.sections = self.parser.sections()

    def read_section(self, section):
        section_dict = {}
        options = self.parser.options(section)
        for option in options:
            try:
                section_dict[option] = self.parser.get(section, option)
            except:
                section_dict[option] = None
                raise("Config reader exception on option %s!" % option)
        return section_dict



def convert_time_unit(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

### Function to handle with argument parsers ####
def str2bool(v):
    """
    Converts string boolean value to boolean value
    :param v: Value as string
    :return: True oder False
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2number(param):
    try:
        number = int(param)
    except ValueError:
        number = float(param)
    return number
