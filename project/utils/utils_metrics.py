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