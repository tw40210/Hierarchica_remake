# from torch.utils.data.dataset import Dataset
# from torch.utils.data import DataLoader
# import torch
import os
import numpy as np
# from utils import read_notefile, note2timestep
import hparam
import random
import matplotlib.pyplot as plt
import pathlib
# import pyworld as pw
import librosa
from tqdm import tqdm
import  mir_eval


ref_intervals, ref_pitches = mir_eval.io.load_valued_intervals('reference.txt')
est_intervals, est_pitches = mir_eval.io.load_valued_intervals('estimate.txt')

mir_eval.transcription.evaluate
print(est_intervals)

#==============
#
#
# dir = "data/train_extension_Process_data/FEAT"
# for file in os.listdir(dir):
#     os.rename(os.path.join(dir, file), os.path.join(dir, f"{file[:-13]}{file[-9:]}"))




#==================pitch augmentation
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
# src_path = "data\\train/TONAS/Deblas"
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





