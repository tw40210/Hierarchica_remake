# from torch.utils.data.dataset import Dataset
# from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from utils import read_notefile, note2timestep
import hparam
import random
import matplotlib.pyplot as plt
import pathlib
import librosa



file_path="data/train/Process_data/FEAT/02-D_ChanoLobato.wav_FEAT.npy"
label_path = "data/train/TONAS/Deblas/02-D_ChanoLobato.notes.Corrected"
wav_path = "data/train/TONAS/Deblas/02-D_ChanoLobato.wav"

y, sr = librosa.load(wav_path, sr=hparam.sr)

label_note = read_notefile(label_path)
label_note, label_pitch = note2timestep(label_note)
label_note = np.array(label_note)
s_label = label_note[:, 0]

feat = np.load(file_path)
norm = feat.max()
feat = feat/norm
norm2 = feat.max()
feat = abs(feat)
feat = np.power(feat, 0.5)
feat = feat[:, :s_label.shape[0]]

plt.figure(figsize=(7, 12))
plt.subplots_adjust(wspace=0, hspace=1)


plt.imshow(feat, vmax=feat.max())
plt.show()

plt.plot(s_label)

plt.show()

