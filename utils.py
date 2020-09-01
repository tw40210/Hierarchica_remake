import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
import librosa
import argparse
import hparam
import matplotlib.pyplot as plt
from typing import Dict, List

def read_notefile(path):
    notes =[]
    with open(path, 'r') as txt:
        lines = txt.readlines()
        lines = lines[1:]
        for line in lines:
            note = list(map(float,line.split(', ')))
            notes.append(note)

    return notes

def note2timestep(notes:List):
    total_timestep= int((notes[-1][0]+notes[-1][1])//0.2+1)
    cur_timestep=0

    for step in range(total_timestep):



if __name__ == '__main__':
    dir = 'data/train/TONAS/Deblas/03-D_Chocolate.notes.Corrected'
    notes = read_notefile(dir)
    print(notes)