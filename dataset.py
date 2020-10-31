from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from utils import read_notefile, note2timestep, silence_label, expand_onoff_label
import hparam
import random
import pathlib

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class mydataset(Dataset):
    def __init__(self, path, f_path, amount=None, augmentation=None):
        self.wav_files = []
        self.labels = []
        self.features = []
        assert len(path) == len(f_path)

        for path_id in range(len(path)):
            self.wav_files = self.wav_files + [os.path.join(path[path_id], file) for file in os.listdir(path[path_id])
                                               if '.wav' in file]
            self.labels = self.labels + [os.path.join(path[path_id], label) for label in os.listdir(path[path_id]) if
                                         '.notes.' in label]
            self.features = self.features + [os.path.join(f_path[path_id], features) for features in
                                             os.listdir(f_path[path_id]) if '_FEAT' in features]
            self.augmentation = augmentation

        if amount:
            while (len(self.wav_files) < amount):
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

        label_path = str(pathlib.Path(self.labels[index]).parent / (
                pathlib.Path(self.features[index]).stem.split('.')[0] + ".notes.Corrected"))

        label_note = read_notefile(label_path)
        # print(self.features[index], "||", label_path)

        label_note, label_pitch = note2timestep(label_note)
        label_note = np.array(label_note)

        label_note = expand_onoff_label(label_note)
        label_pitch = np.array(label_pitch)

        # cut muted tail from feature
        features_full = features_full[:, :label_note.shape[0]]

        # == random sampling
        new_label_note = []
        new_features_full = []
        min_onoff = hparam.onoff
        zero_pad = np.zeros((features_full.shape[0], 9))
        features_full = np.concatenate((zero_pad, features_full), axis=1)
        features_full = np.concatenate((features_full, zero_pad), axis=1)

        if label_note.shape[0] > hparam.randomsample_size - 1:
            for clip_id in range(hparam.randomsample_size):
                start = random.randint(0, label_note.shape[0] - 2)
                if clip_id < min_onoff:
                    while (True):
                        if label_note[start][2] == 1 or label_note[start][4] == 1:
                            if clip_id == 0:
                                new_features_full = np.array(features_full[:, start:start + 19])
                            else:
                                new_features_full = np.concatenate(
                                    (new_features_full, features_full[:, start:start + 19]), axis=1)
                            new_label_note.append(label_note[start])
                            break
                        else:
                            start = random.randint(0, label_note.shape[0] - 2)
                else:
                    if clip_id == 0:
                        new_features_full = np.array(features_full[:, start:start + 19])
                    else:
                        new_features_full = np.concatenate((new_features_full, features_full[:, start:start + 19]),
                                                           axis=1)
                    new_label_note.append(label_note[start])
        else:
            print("Error!! audio is too short!")

        # new_features_full = np.array(new_features_full)
        new_label_note = np.array(new_label_note)

        new_features_full = torch.from_numpy(new_features_full).float()
        # zero_pad = torch.zeros((features_full.shape[0], 9))
        # features_full = torch.cat((zero_pad ,features_full), dim=1) #padding because we need 9 forward and backward
        # features_full = torch.cat(( features_full,zero_pad), dim=1)
        new_features_full = new_features_full.view(9, 174, -1)

        new_features_full = abs(new_features_full)
        new_features_full = np.power(new_features_full / new_features_full.max(),
                                     hparam.gamma_mu)  # normalize &gamma compression

        new_label_note = torch.from_numpy(new_label_note).int()
        if self.augmentation:
            for func in self.augmentation:
                new_features_full = func.process(new_features_full)
        assert new_features_full.shape[2] == hparam.randomsample_size * 19
        assert new_features_full.shape[2] == new_label_note.shape[0] * 19

        return new_features_full, new_label_note

    def __len__(self):
        return len(self.features)


class Random_volume(object):
    def __init__(self, rate, min_range, max_range):
        self.rate = rate
        self.min_range = min_range
        self.max_range = max_range

    def process(self, data):
        if random.random() > self.rate:
            assert self.min_range <= self.max_range
            assert self.min_range >= 0
            scale = random.uniform(self.min_range, self.max_range)
            data *= scale
            return data
        else:
            return data

    ##############################################


if __name__ == '__main__':
    path = 'data/train/TONAS/Deblas'
    f_path = 'data/train/Process_data/FEAT'

    dataloader = DataLoader(mydataset(path, f_path), batch_size=hparam.batch_size, shuffle=True,
                            num_workers=hparam.num_workers)

    for features_full, label_note in dataloader:
        print(features_full.shape, "|||", label_note.shape)
