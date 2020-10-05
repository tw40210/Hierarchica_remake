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



a = np.array([1,0,0,1,0,1])
a = np.tile(a, [10,1])

print(a.shape)