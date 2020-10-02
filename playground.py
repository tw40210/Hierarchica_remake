# from torch.utils.data.dataset import Dataset
# from torch.utils.data import DataLoader
import torch
import os
import numpy as np
# from utils import read_notefile, note2timestep
import hparam
import random
import matplotlib.pyplot as plt
import pathlib

def backward_index(path):
    ''''
    input: posixpath
    :return : posixpath
    '''
    return pathlib.Path(str(path)[:-len(path.stem)])

p = pathlib.Path("/checkpoint")

lib = p/'lib'/'adsd'

print(lib.parent.stem)


