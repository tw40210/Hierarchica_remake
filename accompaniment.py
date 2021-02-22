import numpy as np

chords = [[0, 4, 7], [2, 5, 9], [4, 7, 11], [5, 9, 12], \
          [7, 11, 14], [-3, 0, 4], [-1, 2, 6]]

chord_flow = [[4, 5, 3, 6, 2, 5, 1], [1, 6, 4, 5], [1, 4, 5, 1], [2, 5, 1, 4, 7, 3, 6], [1, 5, 2, 6, 4, 1, 5],
              [4, 3, 6], [4, 3, 2, 1], [6, 5, 4, 3, 2, 6, 4, 1], [1, 2, 3, 4, 3, 2, 5, 1]]


def decompose_octave(pitch):
    return pitch % 12, pitch // 12


def chord_recongnize(interval, pitches):
    octaves = []
    steps = []
    chord_weight = np.zeros(12)
    chord_score = np.zeros(7)

    for pitch in pitches:
        if pitch>0:
            octave, step = decompose_octave(pitch)
            octaves.append(octave)
            steps.append(step)

    for idx, octave in enumerate(octaves):
        chord_weight[int(octave)] += (interval[idx][1] - interval[idx][0]) + 0.2  # give each note a basic score

    for idx, chord in enumerate(chords):
        chord_score[idx] += chord_weight[chord[0] % 12] * 1
        chord_score[idx] += chord_weight[chord[1] % 12] * 0.5
        chord_score[idx] += chord_weight[chord[2] % 12] * 0.5

    if len(steps)>0:
        chord_step = int(np.array(steps).mean()) * 12
    else:
        chord_step=0

    return chord_score.argmax(), chord_step


def tempo_making(interval, onSeqout):
    print(onSeqout)

    tempo_list = np.zeros(8)  # separate  one clip to 8, 1 = downbeat ,2 = sub beat
    num_beats = interval.shape[0]

    if len(onSeqout)>0:
        on_time = interval[onSeqout.argmax()][0] * 100

        if on_time % 25 < 13:  # to get the nearest beat in note
            tempo_list[int(on_time // 25)] = 1
            downbeat = int(on_time // 25)
        elif on_time // 25 < 7:
            tempo_list[int(on_time // 25) + 1] = 1
            downbeat = int(on_time // 25) + 1
        else:
            downbeat=0

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

    note_list.append(
        [0, chord_selected[0] + chord_step - 12, chord_selected[0] + chord_step, chord_selected[1] + chord_step,
         chord_selected[2] + chord_step])

    chord_highlow_flag= True

    for idx in range(8):
        if idx == 0:
            continue
        if tempo_list[idx] == 0:
            if chord_highlow_flag:
                note_list.append([idx, chord_selected[0] + chord_step])
                chord_highlow_flag = not chord_highlow_flag
            else:
                note_list.append([idx, chord_selected[1] + chord_step])
                chord_highlow_flag = not chord_highlow_flag
        elif tempo_list[idx] == 2:
            note_list.append([idx, chord_selected[0] + chord_step])
        elif tempo_list[idx] == 3:
            note_list.append([idx, chord_selected[1] + chord_step, chord_selected[2] + chord_step])
        elif tempo_list[idx] == 1:
            note_list.append(
                [idx, chord_selected[0] + chord_step, chord_selected[1] + chord_step, chord_selected[2] + chord_step])


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
    return chord_prob_table[chord1, chord2, :].argmax()
