from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from utils import read_notefile, note2timestep
import hparam
import random

dir1='data/train/Process_data/FEAT\\43-M1_ElChocolate.wav_FEAT.npy'
dir2='data/train/Process_data/FEAT\\29-M1_MSimon-Martinete.wav_FEAT.npy'

a1=np.load(dir1)
a2=np.load(dir2)

print(a1)
