import random
import numpy
import torch
import collections


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.nanmean(array)
    d["std"] = numpy.nanstd(array)
    d["min"] = numpy.nanmin(array)
    d["max"] = numpy.nanmax(array)
    return d
