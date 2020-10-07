from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from utils import read_notefile, note2timestep, silence_label, expand_onoff_label
import hparam
import random

SEED=0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

class mydataset(Dataset):
    def __init__(self, path, f_path, amount =None):
        self.wav_files = [os.path.join(path,file ) for file in os.listdir(path) if '.wav' in file]
        self.labels = [os.path.join(path,label ) for label in os.listdir(path) if'.notes.' in label]
        self.features = [os.path.join(f_path,features )  for features in os.listdir(f_path) if '_FEAT' in features]

        if amount:
            while(len(self.wav_files)<amount):
                self.wav_files = self.wav_files + self.wav_files
                self.labels = self.labels + self.labels
                self.features = self.features + self.features

            self.wav_files = self.wav_files[:amount]
            self.labels = self.labels[:amount]
            self.features = self.features[:amount]

        print(len(self.features))

    def __getitem__(self, index):
        # dir = 'data/train/Process_data/FEAT\\43-M1_ElChocolate.wav_FEAT.npy'
        # features_full = np.load(dir)
        features_full = np.load(self.features[index])

        label_note = read_notefile(self.labels[index])
        label_note, label_pitch = note2timestep(label_note)
        label_note = np.array(label_note)

        label_note = expand_onoff_label(label_note)
        label_pitch = np.array(label_pitch)

        # cut muted tail from feature
        features_full = features_full[:, :label_note.shape[0]]


        #== random sampling



        if features_full.shape[1]>hparam.randomsample_size-1:
            onoff_exist=0
            while(True):
                start = random.randint(0, features_full.shape[1]-hparam.randomsample_size-19)
                new_features_full = features_full[:,start:start+hparam.randomsample_size+18]
                new_label_note = label_note[start+9:start+int(hparam.randomsample_size)+9]
                for note in new_label_note:
                    if(note[2]==1 or note[4]==1):
                        onoff_exist+=1
                if(onoff_exist>2):
                    break
                onoff_exist=0



        else:
            # print(features_full.shape[1], label_note.shape[0], self.labels[index])
            zero_pad = np.zeros((features_full.shape[0], hparam.randomsample_size - features_full.shape[1]))
            new_features_full = np.concatenate((features_full, zero_pad), axis=1)
            zero_pad = silence_label((hparam.randomsample_size - label_note.shape[0], label_note.shape[1]))
            new_label_note = np.concatenate((label_note, zero_pad), axis=0)

        new_features_full= torch.from_numpy(new_features_full).float()
        # zero_pad = torch.zeros((features_full.shape[0], 9))
        # features_full = torch.cat((zero_pad ,features_full), dim=1) #padding because we need 9 forward and backward
        # features_full = torch.cat(( features_full,zero_pad), dim=1)
        new_features_full= new_features_full.view(3,522,-1)
        new_label_note = torch.from_numpy(np.array(new_label_note)).int()

        assert new_features_full.shape[2]==hparam.randomsample_size+18


        return new_features_full, new_label_note

    def __len__(self):
        return len(self.features)


    ##############################################
if __name__ == '__main__':
    path = 'data/train/TONAS/Deblas'
    f_path ='data/train/Process_data/FEAT'

    dataloader = DataLoader(mydataset(path, f_path), batch_size = hparam.batch_size, shuffle=True, num_workers=hparam.num_workers)

    for features_full, label_note in dataloader:
        print(features_full.shape, "|||", label_note.shape)