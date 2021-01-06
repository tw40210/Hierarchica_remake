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
import crepe


# !/usr/bin/python
# -*- coding: UTF-8 -*-

dir = "data/train/train_extension_Process_data/FEAT/"

for file in os.listdir(dir):
    if ".wav" not in file :
        os.rename(dir+file, dir+file[:-9]+".wav"+file[-9:])
#
# y, sr = librosa.load("data/train/TONAS/Deblas/01-D_AMairena.wav")
# time,frequency,confidence,activation = crepe.predict(y,sr,viterbi=True)
# print(frequency)
#

#+==============
# import threading
# import time
#
#
# class myThread(threading.Thread):
#     def __init__(self, threadID, name, counter):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#         self.counter = counter
#
#     def run(self):
#         print(        "Starting " + self.name)
#
#         # 获得锁，成功获得锁定后返回True
#         # 可选的timeout参数不填时将一直阻塞直到获得锁定
#         # 否则超时后将返回False
#         threadLock.acquire()
#         print_time(self.name, self.counter, 3)
#         # 释放锁
#         threadLock.release()
#
#
# def print_time(threadName, delay, counter):
#     while counter:
#         time.sleep(delay)
#         print("%s: %s" % (threadName, time.ctime(time.time())))
#
#         counter -= 1
#
#
# threadLock = threading.Lock()
# threads = []
#
# # 创建新线程
# thread1 = myThread(1, "Thread-1", 1)
# thread2 = myThread(2, "Thread-2", 2)
#
# # 开启新线程
# thread1.start()
# thread2.start()
#
# # 添加线程到线程列表
# threads.append(thread1)
# threads.append(thread2)
#
# # 等待所有线程完成
# for t in threads:
#     t.join()
# print("Exiting Main Thread")



#
# ref_intervals, ref_pitches = mir_eval.io.load_valued_intervals('reference.txt')
# est_intervals, est_pitches = mir_eval.io.load_valued_intervals('estimate.txt')
#
# mir_eval.transcription.evaluate
# print(est_intervals)


#==============
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




