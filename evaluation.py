import numpy as np
import librosa
from preprocess import output_feature_extraction_nosave
import hparam
import os
from utils import signal_sampletest_stream, rawout2interval_picth, get_Resnet,read_sheetlabel
import torch
import accompaniment


def soloCliptest(file_path,model, RATE=16000):
    buffer = np.zeros((hparam.FEAT_freqbin_num * hparam.FEAT_channel, hparam.FEAT_pastpad + hparam.FEAT_futurepad))
    wavform_buffer = np.zeros((int(RATE * hparam.timestep * hparam.FEAT_futurepad)))
    data_float, sr = librosa.load(file_path, sr=RATE)
    SN_SIN_ZN = output_feature_extraction_nosave(data_float, window_size=[768, 372, 186])
    buffer[:,hparam.FEAT_pastpad:hparam.FEAT_pastpad + hparam.FEAT_futurepad] = SN_SIN_ZN[:, 0:hparam.FEAT_pastpad] # repad FEAT to get a zero head and tail
    SN_SIN_ZN= np.concatenate((SN_SIN_ZN[:,:-hparam.FEAT_futurepad], np.zeros((hparam.FEAT_freqbin_num * hparam.FEAT_channel, hparam.FEAT_futurepad))), axis=1)

    record, buffer = signal_sampletest_stream(SN_SIN_ZN,past_buffer=buffer, model=model, channel=hparam.FEAT_channel)
    # est_intervals, _, _, _, _, _, onstart_flag = Smooth_sdt6(record, realtime=True, onstart_flag= onstart_flag)

    padding_data_float = data_float[int(-RATE*hparam.timestep*(hparam.FEAT_pastpad)):]
    data_float = np.concatenate((wavform_buffer, data_float[:int(-RATE*hparam.timestep*(hparam.FEAT_pastpad))]), axis=0) # adjust wav signal to match label
    wavform_buffer = padding_data_float
    librosa.output.write_wav(f"wav_check/solotest.wav", data_float, sr=RATE)
    interval, pitches, onstart_flag, onSeqout = rawout2interval_picth(record, data_float, sr=RATE, onstart_flag=False)
    return interval, pitches

def gt_midimatch(gt_txtpath, est_npypath):
    est_nparray = np.load(est_npypath,allow_pickle=True)
    gt_nparray = np.array(read_sheetlabel(gt_txtpath))

    beats_total =0
    beats_correct =0
    chord_total=0
    chord_est_correct=0
    chord_gt_correct=0

    if est_nparray.shape[0]> gt_nparray.shape[0]:
        midilength = gt_nparray.shape[0]
    else:
        midilength = est_nparray.shape[0]

    for idx in range(midilength):
        for beat in est_nparray[idx][0]:
            if beat in gt_nparray[idx][0]:
                beats_correct+=1
            beats_total+=1

        # for note_pitch in gt_nparray[idx][2]:
        #     aa = np.array(accompaniment.chords[accompaniment.chord_index_inv[gt_nparray[idx][1]]]) % 12
        #     if int(note_pitch) in np.array(accompaniment.chords[ accompaniment.chord_index_inv[gt_nparray[idx][1]]  ]) % 12: # check if pitch in selected chord
        #         chord_correct+=1

        for note_pitch in gt_nparray[idx][2]:
            aa = np.array(accompaniment.chords[accompaniment.chord_index_inv[est_nparray[idx][1]]]) % 12
            b = note_pitch
            if int(note_pitch) in np.array(accompaniment.chords[ accompaniment.chord_index_inv[est_nparray[idx][1]]  ]) % 12: # check if pitch in selected chord
                chord_est_correct+=1
            if int(note_pitch) in np.array(accompaniment.chords[ accompaniment.chord_index_inv[gt_nparray[idx][1]]  ]) % 12: # check if pitch in selected chord
                chord_gt_correct+=1

            chord_total+=1


    return beats_correct/beats_total, chord_est_correct/chord_total,chord_gt_correct/chord_total

def output_integration(midi_dir, wav_dir):
    from midi2audio import FluidSynth
    import shutil
    midiout_dir = "midi_check/midi_wav"
    finalout_dir ="offline_output"

    shutil.rmtree("midi_check/midi_wav")
    os.makedirs("midi_check/midi_wav", exist_ok=True)
    fs = FluidSynth()
    midiwav_y = np.zeros(0)
    wav_y = np.zeros(0)

    for midifile in os.listdir(midi_dir):
        if ".mid" in midifile:
            fs.midi_to_audio(f'{midi_dir}/{midifile}', f'midi_check/midi_wav/{midifile[:-4]}.wav')


    midipath_list = os.listdir(midiout_dir)
    wavpath_list = os.listdir(wav_dir)

    def pathsort(path):
        return int(path.split(".")[0])

    midipath_list.sort(key=pathsort)
    wavpath_list.sort(key=pathsort)



    for midiwav in midipath_list:
        y, sr  = librosa.load(f'{midiout_dir}/{midiwav}', sr=hparam.sr)
        midiwav_y = np.concatenate((midiwav_y, y), axis=0)

    for wav_file in wavpath_list:
        y, sr  = librosa.load(f'{wav_dir}/{wav_file}', sr=hparam.sr)
        wav_y = np.concatenate((wav_y, y), axis=0)

    wav_y/=wav_y.max()*4


    if wav_y.shape[0]> midiwav_y.shape[0]:
        wav_y = wav_y[:midiwav_y.shape[0]]
    else:
        midiwav_y = midiwav_y[:wav_y.shape[0]]

    wav_y = wav_y+midiwav_y*10

    librosa.output.write_wav(f"{finalout_dir}/finaltest.wav", wav_y,  sr=hparam.sr)



if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = get_Resnet(hparam.FEAT_channel).to(device)
    # model.load_state_dict(torch.load("standard_checkpoint/960_1030perform077.pth"))
    # print("load OK")
    # interval, pitches = soloCliptest(file_path="5.wav", model= model)
    #
    # print("interval: ", interval)
    # print("fin")

    gt_path = "TEST/Jay Chou_Sunny Day_vocal.txt"
    est_path = "midi_record/Jay Chou_Sunny Day_vocal.npy"

    print(gt_midimatch(gt_path, est_path))


    # output_integration("midi_check", "wav_check")
