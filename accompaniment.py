import numpy as np
import hparam
import random

chords = [[0, 4, 7], [2, 5, 9], [4, 7, 11], [5, 9, 12], \
          [7, 11, 14], [-3, 0, 4], [-1, 2, 6]]

chord_flow = [[4, 5, 3, 6, 2, 5, 1], [1, 6, 4, 5], [1, 4, 5, 1], [2, 5, 1, 4, 7, 3, 6], [1, 5, 2, 6, 4, 1, 5],
              [4, 3, 6], [4, 3, 2, 1], [6, 5, 4, 3, 2, 6, 4, 1], [1, 2, 3, 4, 3, 2, 5, 1]]

chord_index = {0: "C", 1: "D", 2: "E", 3: "F", 4: "G", 5: "A", 6: "B"}
chord_index_inv = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
eight_to_twelve_octave={0:-1, 1:0, 2:2, 3:4, 4:5, 5:7, 6:9, 7:11}


def decompose_octave(pitch, tone):
    return pitch % 12, (pitch + tone) // 12


def chord_recongnize(interval, pitches):
    octaves = []
    steps = []
    chord_weight = np.zeros(12)
    chord_score = np.zeros(7)
    tone = hparam.song_tone
    if len(pitches) > 0:
        pitches -= tone

    for pitch in pitches:
        if pitch > -tone:  # original pitch >0
            octave, step = decompose_octave(pitch, tone)
            octaves.append(octave)
            steps.append(step)

    for idx, octave in enumerate(octaves):
        chord_weight[int(octave)] += (interval[idx][1] - interval[idx][0]) + 0.2  # give each note a basic score

    for idx, chord in enumerate(chords):
        chord_score[idx] += chord_weight[chord[0] % 12] * 1
        chord_score[idx] += chord_weight[chord[1] % 12] * 0.5
        chord_score[idx] += chord_weight[chord[2] % 12] * 0.5

    if len(steps) > 0:
        chord_step = int(np.array(steps).mean()) * 12
    else:
        chord_step = 0

    return chord_score.argmax(), chord_step


def tempo_making(interval, onSeqout):
    print(onSeqout)

    tempo_list = np.zeros(8)  # separate  one clip to 8, 1 = downbeat ,2 = sub beat
    num_beats = interval.shape[0]
    note_length = (60 / hparam.song_bpm) / 2 #8-th note as a unit

    if len(onSeqout) > 0:
        on_time = interval[onSeqout.argmax()][0]
        a = on_time % note_length
        b = (on_time // note_length)*2

        if (on_time // note_length)*2 >= tempo_list.shape[0] - 2 and on_time % note_length >= note_length / 2:
            tempo_list[tempo_list.shape[0] - 1] = 1
            downbeat = tempo_list.shape[0] - 1
        elif on_time % note_length < note_length / 2 :  # to get the nearest beat in note
            tempo_list[int((on_time // note_length)*2)] = 1
            downbeat = int((on_time // note_length)*2)
        elif on_time % note_length >= note_length / 2:  # to get the nearest beat in note
            tempo_list[int((on_time // note_length)*2)+2] = 1
            downbeat = int((on_time // note_length)*2)+2
        else:
            downbeat = 0

        if num_beats < 3:
            if downbeat < 4:
                tempo_list[6] = 2
                tempo_list[7] = 3
            else:
                tempo_list[1] = 2
                tempo_list[2] = 3

    return tempo_list


def note_making(tempo_list, chord_idx, chord_step):
    chord_selected = chords[chord_idx]
    note_list = []  # struct [idx, pitch1, pitch2 ... ] keep to end fo music
    if hparam.song_tone > 5:
        chord_step -= 12  # offset tone complimentation

    note_list.append(
        [0, chord_selected[0] + chord_step - 12 + hparam.song_tone, chord_selected[0] + chord_step + hparam.song_tone,
         chord_selected[1] + chord_step + hparam.song_tone,
         chord_selected[2] + chord_step + hparam.song_tone])

    chord_highlow_flag = True

    for idx in range(8):
        if idx == 0:
            continue
        if tempo_list[idx] == 0:
            if tempo_list[idx - 1] > 0:
                continue
            if chord_highlow_flag:
                note_list.append([idx, chord_selected[0] + chord_step + hparam.song_tone])
                chord_highlow_flag = not chord_highlow_flag
            else:
                note_list.append([idx, chord_selected[1] + chord_step + hparam.song_tone])
                chord_highlow_flag = not chord_highlow_flag
        elif tempo_list[idx] == 2:
            note_list.append([idx, chord_selected[0] + chord_step + hparam.song_tone,
                              chord_selected[1] + chord_step + hparam.song_tone])
        elif tempo_list[idx] == 3:
            note_list.append([idx, chord_selected[1] + chord_step + hparam.song_tone,
                              chord_selected[2] + chord_step + hparam.song_tone])
        elif tempo_list[idx] == 1:
            note_list.append(
                [idx, chord_selected[0] + chord_step + hparam.song_tone,
                 chord_selected[1] + chord_step + hparam.song_tone, chord_selected[2] + chord_step + hparam.song_tone])

    return note_list


def chord_probability():
    chord_prob_table = np.zeros((7, 7, 7))
    count = 0
    for flow in chord_flow:
        for idx in range(len(flow)):
            chord_prob_table[
                flow[idx - 2] - 1, flow[idx - 1] - 1, flow[idx] - 1] += 3  # previous two chord are all right
            chord_prob_table[:, flow[idx - 1] - 1, flow[idx] - 1] += 1
            count += 11

    chord_prob_table = chord_prob_table / count
    print(chord_prob_table)
    np.save("chord_probability.npy", chord_prob_table)

    return 0


def chord_predict(chord1, chord2, chord_prob_table):
    current_list = chord_prob_table[chord1, chord2, :]
    current_list/=current_list.sum()
    rand = random.random()

    count=0
    for idx, prob in enumerate(current_list) :
        if rand< count+prob:
            return idx
        else:
            count+=prob

    raise Exception("No chord chosen!")