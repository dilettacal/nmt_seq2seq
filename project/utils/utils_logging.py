import os

import dill
import numpy as np
import torch


class Logger(object):
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


class MetricPlot(object):
    def __init__(self, path):
        self.path = path

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
            return "Plot saved: {}".format(save_path)

        elif isinstance(metric, list):
            ### this plots one metric
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel('epoch')
            plt.plot(metric, label=ylabel, color="b")
            plt.legend(loc='upper right')
            plt.savefig(save_path, format="png", dpi=500)
            plt.close()
            return "Plot saved: {}".format(save_path)

        else:
            raise Exception("Provide metric either as list or as dictionary containing 2 metrics.")
