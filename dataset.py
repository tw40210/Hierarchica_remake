from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from utils import read_notefile
import hparam

class train_set(Dataset):
    def __init__(self, path, f_path):
        self.wav_files = [os.path.join(path,file ) for file in os.listdir(path) if '.wav' in file]
        self.labels = [os.path.join(path,label ) for label in os.listdir(path) if'.notes.' in label]
        self.features = [os.path.join(f_path,features )  for features in os.listdir(f_path) if '_FEAT' in features]


    def __getitem__(self, index):
        features_full = np.load(self.features[index])
        label_note = read_notefile(self.labels[index])

        features_full = features_full[:,:250]




        return features_full, label_note

    def __len__(self):
        return len(self.features)


    ##############################################
if __name__ == '__main__':
    path = 'data/train/TONAS/Deblas'
    f_path ='data/train/Process_data/FEAT'

    dataloader = DataLoader(train_set(path, f_path), batch_size = hparam.batch_size, shuffle=True, num_workers=hparam.num_workers)

    for features_full, label_note in dataloader:
        print(features_full, label_note)