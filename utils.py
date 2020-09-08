import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
import librosa
import argparse
import hparam
import matplotlib.pyplot as plt
from typing import Dict, List
import os


def read_notefile(path, limit_len=None):
    notes = []
    with open(path, 'r') as txt:
        lines = txt.readlines()
        lines = lines[1:]
        for line in lines:
            note = list(map(float, line.split(', ')))
            notes.append(note)

    return notes


def note2timestep(notes: List):
    timestep = []
    pitch=[]
    tail=0
    end_tail=0
    for idx, note in enumerate(notes):
        status = [1, 0, 0, 1, 0, 1]  # S, A, O, -O, X, -X
        while (len(timestep) < note[0] // 0.02):
            timestep.append(status)
            pitch.append(0)

        if idx > 0:
            if note[0]-end_tail<1e-4:
                timestep[-1] = [0, 1, 1, 0, 1, 0]
                pitch[-1]=(note[2])
            else:
                status = [0, 1, 1, 0, 0, 1]
                timestep.append(status)
                pitch.append(note[2])
        else:
            status = [0, 1, 1, 0, 0, 1]
            timestep.append(status)
            pitch.append(note[2])

        # tail = note[0] // 0.02 * 0.02 + 0.02
        tail=len(timestep)*0.02
        end_tail = (note[0]+note[1])// 0.02 * 0.02 + 0.02
        status = [0, 1, 0, 1, 0, 1]
        ccc = ((note[0] + note[1] - tail) / 0.02)
        for _ in range(int((note[0] + note[1] - tail+1e-4) // 0.02)):
            timestep.append(status)
            pitch.append(note[2])

        status = [0, 1, 0, 1, 1, 0]
        timestep.append(status)
        pitch.append(note[2])
        # print(len(timestep), len(pitch))

    return timestep, pitch


if __name__ == '__main__':
    for file in os.listdir('data/train/TONAS/Deblas/'):
        if '.notes.Corrected' in file:
            dir = f'data/train/TONAS/Deblas/{file}'
            notes = read_notefile(dir)
            aa,pp = note2timestep(notes)

            print(((notes[-1][0]+notes[-1][1]+1e-4)//0.02+1)*0.02, len(aa)*0.02,file)
            assert ((notes[-1][0]+notes[-1][1]+1e-4)//0.02+1)*0.02==len(aa)*0.02

            aa = np.array(aa)
            pp = np.array(pp)


    # dir = f'data/train/TONAS/Deblas/52-M1_ManueldeAngustias.notes.Corrected'
    # notes = read_notefile(dir)
    # aa,pp = note2timestep(notes)
    # aa = np.array(aa)
    # pp = np.array(pp)