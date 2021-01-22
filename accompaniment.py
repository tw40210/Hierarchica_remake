import numpy as np

chords = [[0, 4, 7], [2, 5, 9], [4, 7, 11], [5, 9, 12], \
          [7, 11, 14], [-3, 0, 4], [-1, 2, 6]]

tempo_open=[['half', 0, [1,3],[0,0]],['eighth', 0, [1], [0]],['quarter', 1, [1,1], [0,1]]]
tempo_run=[[['quarter', 0, [1,2,3], [0,0,0]], ['quarter', 0, [1,3], [0,0]],['quarter', 0, [1,2,3], [0,0,0]], ['quarter', 0, [1,3], [0,0]]],\
           [['quarter', 0, [1,2,3], [0,0,0]],['eighth', 0, [2], [0]],['eighth', 0, [3], [0]],['eighth', 0, [2], [0]],['eighth', 0, [1], [0]],['quarter', 0, [1,3], [0,0]]],\
           [['quarter', 0, [1,2,3], [0,0,0]],['eighth', 0, [2], [0]],['eighth', 0, [3], [0]],['quarter', 0, [2], [0]],['quarter', 0, [1,3], [0,0]]],\
          [['eighth', 0, [1,2,3],[0,0,0]],['eighth', 0, [1,2],[0,0]],['16th', 0, [2], [0]],['16th', 0, [3], [0]],['eighth', 0, [2,3], [0,-1]],['16th', 0, [2], [0]],['16th', 0, [1], [0]],['eighth', 0, [1,2], [0,0]],['16th', 0, [1,3], [0,0]],['16th', 0, [1,2], [0,0]],['eighth', 0, [1,1], [0,1]]]]
tempo_end=[['half', 0, [1,2,3],[0,0,0]],['half', 0, [1,3,1],[0,-1,-1]]]

def decompose_octave(pitch):
    return pitch // 12, pitch % 12


def chord_recongnize(interval, pitches):
    octaves = []
    steps = []
    chord_weight = np.zeros(12)
    chord_score = np.zeros(7)

    for pitch in pitches:
        octave, step = decompose_octave(pitch)
        octaves.append(octave)
        steps.append(step)

    for idx, step in enumerate(steps):
        chord_weight[int(step)] += (interval[idx][1] - interval[idx][0]) + 0.2  # give each note a basic score

    for idx, chord in enumerate(chords):
        chord_score[idx] += chord_weight[chord[0]%12] * 1
        chord_score[idx] += chord_weight[chord[1]%12] * 0.5
        chord_score[idx] += chord_weight[chord[2]%12] * 0.5

    return chord_score.argmax()

def beat_dense(interval):
    pass



def tempo_predict(interval):
    pass

