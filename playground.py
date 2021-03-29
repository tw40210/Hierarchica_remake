# from torch.utils.data.dataset import Dataset
# from torch.utils.data import DataLoader
# import torch
import os
import numpy as np
from utils import read_notefile, note2timestep, interval2pitch_in_note, get_Resnet
import hparam
import random
import matplotlib.pyplot as plt
import pathlib
# import pyworld as pw
import librosa
from tqdm import tqdm
import mir_eval

from accompaniment import chord_probability, chord_predict
import math
from resnest import resnest50
import torch
import accompaniment
from thop import profile
# import torch.nn as nn
from torchstat import stat


input = torch.randn(10, 9, 174, 19)
# input = torch.randn(10, 3, 224, 224)
# model = resnest50(channel=9, num_classes=6)
model= get_Resnet(channel=hparam.FEAT_channel, is_simplified=True)
PATH = "resnest_model.pth"
# torch.save(model.state_dict(), PATH)

x = model(input)

# macs, params = profile(model, inputs=(input, ))

stat(model, (9, 174, 19))

# print(macs, params)

print( np.array(accompaniment.chords[6])%12)


# ##=================
# for midifile in os.listdir("midi_check"):
#     if ".mid" in midifile:
#         os.rename(f"midi_check/{midifile}", f"midi_check/{midifile[:-5]}.mid")

# ==============
#
#
# dir = "data/train_extension_Process522/FEAT"
# for file in os.listdir(dir):
#     os.rename(os.path.join(dir, file), os.path.join(dir, f"{file[:-13]}{file[-9:]}"))


# ==================pitch augmentation
# def write_label(src_path, tar_path, action, scale):
#     with open(src_path, 'r') as src_txt:
#
#         all = []
#         for line in src_txt.readlines():
#             this_line = []
#             for i in line.split(", "):
#                 this_line.append(float(i))
#             all.append(this_line)
#
#         with open(tar_path, 'w') as tar_txt:
#             for line in all:
#                 for index, item in enumerate(line):
#                     if index == 2:
#                         if action == "dw":
#                             tar_txt.write(str(item - scale))
#
#                         else:
#                             tar_txt.write(str(item + scale))
#                     else:
#                         tar_txt.write(str(item))
#
#                     if index < len(line) - 1:
#                         tar_txt.write(", ")
#                 tar_txt.write("\n")
#
#     print(tar_path)
#
#
# src_path = "data/train/TONAS/Deblas"
# tar_path = "data/train/train_extension"
# fs= 44100
# actions = ["dw", "up"]
# scales = [ 1, 2]
# for file in tqdm(os.listdir(src_path)):
#         if ".wav" in file:
#             for action in actions:
#                 for scale in scales:
#                     wavfile= os.path.join(src_path, file)
#                     labelfile = os.path.join(src_path, f"{file[:-4]}.notes.Corrected")
#
#                     tarwavfile = os.path.join(tar_path, f"{file[:-4]}_{action}{scale}.wav")
#                     tarlabelfile = os.path.join(tar_path,f"{file[:-4]}_{action}{scale}.notes.Corrected")
#
#                     y, sr = librosa.load(wavfile, sr=fs)
#                     y = np.array(y, dtype="double")
#                     f0, sp, ap = pw.wav2world(y, fs)
#
#                     if action=="dw":
#                         f0 = f0 / np.power(2, 1 / (12/scale))
#                     else:
#                         f0 = f0 * np.power(2, 1 / (12 / scale))
#
#                     new_y = pw.synthesize(f0, sp, ap, fs)
#                     new_y = np.array(new_y, dtype="float")
#                     librosa.output.write_wav(tarwavfile, new_y, sr=fs)
#
#                     write_label(labelfile, tarlabelfile, action, scale)
#
