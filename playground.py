# from torch.utils.data.dataset import Dataset
# from torch.utils.data import DataLoader
# import torch
import os
import numpy as np
from utils import read_notefile, note2timestep, interval2pitch_in_note
import hparam
import random
import matplotlib.pyplot as plt
import pathlib
# import pyworld as pw
import librosa
from tqdm import tqdm
import mir_eval
import crepe
from accompaniment import chord_probability, chord_predict
import midi2audio
from midi2audio import FluidSynth
import math

lat_long1 =np.array([20, 123])
theta, d = 90* math.pi / 180, 50
R = 6357
lat_long2 = [21, 124]

lat_long1 = lat_long1 * math.pi / 180

lat2 = math.asin(
    (math.sin(lat_long1[0]) * math.cos(d / R)) + (math.cos(lat_long1[0]) * math.cos(theta) * math.sin(d / R)))
long2 = lat_long1[1] + math.atan((math.cos(lat_long1[0]) * math.sin(theta) * math.sin(d / R)) / (
            math.cos(d / R) - (math.sin(lat_long1[0]) * math.sin(lat2))))



print(lat2* 180  / math.pi,long2* 180  / math.pi )
print(lat2,long2 )
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
