import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from utils import read_notefile, note2timestep
import hparam
import random
from tqdm import tqdm
from dataset import train_set
from model import ResNet, BasicBlock

path = 'data/train/TONAS/Deblas'
f_path = 'data/train/Process_data/FEAT'



def train():

    dataloader = DataLoader(train_set(path, f_path), batch_size=hparam.batch_size, shuffle=True,
                            num_workers=hparam.num_workers)
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    num_fout = model.conv1.out_channels
    model.conv1 = nn.Conv2d(3, num_fout, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                               bias=False)
    model.fc = nn.Linear(model.fc.in_features*7, 6)

    model.avgpool = nn.AvgPool2d(kernel_size=(17, 1), stride=1, padding=0)

    for features_full, label_note in tqdm(dataloader) :


if __name__ == '__main__':
