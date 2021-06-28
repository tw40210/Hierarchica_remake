import pyaudio
import matplotlib
import numpy as np
from preprocess import output_feature_extraction_nosave
import time
from utils import signal_sampletest_stream, get_Resnet, Smooth_sdt6, rawout2interval_picth
import torch
import hparam
import librosa
from midiutil.MidiFile import MIDIFile
from io import BytesIO
import pygame
import pygame.mixer
from time import sleep
import threading
import accompaniment
from accompaniment import tempo_making, note_making, chord_recongnize, chord_predict
import os
from evaluation import output_integration


import pygame
import pygame.mixer
from time import sleep

pygame.init()
pygame.mixer.init()

song_bpm = hparam.song_bpm
matplotlib.use('TKAgg', warn=False, force=True)
print("Switched to:", matplotlib.get_backend())
RATE = 16000
CHUNK = int((60 / song_bpm) * RATE * 2)
FORMAT = pyaudio.paFloat32
CHANNELS = 1
bar_limit = (CHUNK / RATE)-0.3
is_offline = False
do_clean=True
if do_clean:
    wav_folder="wav_check"
    for file in os.listdir(wav_folder):
        if ".wav" in file:
                os.remove(f"{wav_folder}/{file}")

    midi_folder="midi_check"
    for file in os.listdir(midi_folder):
        if ".mid" in file:
                os.remove(f"{midi_folder}/{file}")


if is_offline:
    wavfile_path = "TEST/forgetme_2.wav"
    wav_signal, sr = librosa.load(wavfile_path, sr=RATE)
    wav_signal = np.concatenate((wav_signal[-CHUNK*2:], wav_signal), axis=0)

chord_index = {0: "C", 1: "D", 2: "E", 3: "F", 4: "G", 5: "A", 6: "B"}

SHOOT_SOUND = pygame.mixer.Sound('data/bass.wav')
SHOOT_SOUND.set_volume(0.04)

global mide_file
global play_lock
load_flag = False


def midi_record(tempo_list, chord_next, midirecord):
    beats = []
    for idx in range(tempo_list.shape[0]):
        if tempo_list[idx] == 1:   # 16-th to 32-th to fit gt_midi
            beats.append(idx*2)
    midirecord.append([beats, accompaniment.chord_index[chord_next]])


def play_midi():
    global play_lock
    while True:
        if play_lock:
            pygame.mixer.music.load(mide_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                sleep(0.01)
            # SHOOT_SOUND.play()
            play_lock = False
            print("Done!")
        else:
            sleep(0.02)


def waitBuffer(minBuffer):
    while stream.get_read_available() < minBuffer:
        print(stream.get_read_available())
        sleep(0.01)


def clearBuffer(stream, CHUNK):
    while stream.get_read_available() > CHUNK/2:
        stream.read(CHUNK)

    # stream.read( int(CHUNK*0.8)) # rest 0.2 length buffer


def convert_seconds_to_quarter(time_in_sec, bpm):
    quarter_per_second = (bpm / 60)
    time_in_quarter = time_in_sec * quarter_per_second
    return time_in_quarter


def create_MIDI(note_list, count=0, volume_para=1):
    volume_para = max(min(volume_para**0.5, 2), 1)
    memFile = BytesIO()
    MyMIDI = MIDIFile(1)
    track = 0
    time = 0
    channel = 0
    pitch = 60
    duration = 1
    volume = int(30 * volume_para)
    end_time = CHUNK / RATE
    bpm = song_bpm
    MyMIDI.addTrackName(track, time, "Sample Track")
    MyMIDI.addTempo(track, time, bpm)

    if not is_offline:
        delay = 0  # delay mainly by pygame load& play can't avoid so far because load will stop music
    else:
        delay=0

    for note in note_list:
        beat_pos = note[0]
        beat_time=beat_pos * (end_time/8)
        time = convert_seconds_to_quarter(beat_time, bpm)
        duration = convert_seconds_to_quarter(end_time - beat_time +delay, bpm)
        new_observer = end_time - beat_time +delay
        for idx in range(len(note)):
            if idx == 0:
                continue
            if len(note)>2:
                MyMIDI.addNote(track, channel, int(note[idx]), time, duration, volume)
            else:
                MyMIDI.addNote(track, channel, int(note[idx]), time, duration, int(volume*0.8))

    if not is_offline:
        MyMIDI.writeFile(memFile)
    else:
        with open(f"midi_check/{count}.mid", 'wb') as binfile:
            MyMIDI.writeFile(binfile)

    memFile.seek(0)

    return memFile


p = pyaudio.PyAudio()

if not is_offline:
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=int(CHUNK),
        # stream_callback = callbackln
    )

    stream.start_stream()

count = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_Resnet(hparam.FEAT_channel, is_simplified=True).to(device)
model.load_state_dict(torch.load("standard_checkpoint/3150_on079l3dnesec8.pth"))
print("load OK")

buffer = np.zeros((hparam.FEAT_freqbin_num * hparam.FEAT_channel, hparam.FEAT_pastpad + hparam.FEAT_futurepad))
wavform_buffer = np.zeros((int(RATE * hparam.timestep * hparam.FEAT_futurepad)))
# print("keyin anython to start")
# input()  # pause

midirecord = []
midi_playing = False
onstart_flag = False
chord_record = np.zeros(2, dtype=int)  # start with C, C
old_note_list = [[0, 48, 51, 54]]
volume_ori_para =1
volume_cur_para =1

chord_prob_table = np.load("chord_probability.npy")
chord_next = 0

time_cost=0
iter_num=0
max_time=[]


music_start = time.time()
while True:

    count += 1
    if not is_offline:
        if count < 4:
            clearBuffer(stream, CHUNK)
            # waitBuffer(int(CHUNK * 0.2))
        print("buffer to read: ", stream.get_read_available())
        if(count==2):
            data_load = stream.read(CHUNK)
            data_load2 = stream.read(CHUNK)
        else:
            data_load = data_load2
            data_load2 = data_cur
        data_cur = stream.read(CHUNK)
    else:
        if count * CHUNK > len(wav_signal):
            midirecord = np.array(midirecord)
            np.save(f"midi_record/{wavfile_path.split('/')[-1][:-4]}.npy", midirecord)
            print("Signal finished!")
            break

        data_load = wav_signal[(count - 2) * CHUNK:(count - 1)  * CHUNK]
        data_cur = wav_signal[(count) * CHUNK:(count +1)  * CHUNK]

    start = time.time()
    data_float = np.fromstring(data_load, 'Float32')
    data_float_cur = np.fromstring(data_cur, 'Float32')

    if count==5:
        volume_ori_para = abs(data_float).mean()

    volume_cur_para = abs(data_float).mean()

    SN_SIN_ZN = output_feature_extraction_nosave(data_float, window_size=[768, 372, 186])
    record, buffer = signal_sampletest_stream(SN_SIN_ZN, past_buffer=buffer, model=model, channel=hparam.FEAT_channel)

    padding_data_float = data_float[int(-RATE * hparam.timestep * (hparam.FEAT_pastpad)):]
    data_float = np.concatenate((wavform_buffer, data_float[:int(-RATE * hparam.timestep * (hparam.FEAT_pastpad))]),
                                axis=0)  # adjust wav signal to match label
    wavform_buffer = padding_data_float
    librosa.output.write_wav(f"wav_check/{count}.wav", data_float_cur, sr=RATE)
    interval, pitches, onstart_flag, onSeqout = rawout2interval_picth(record, data_float, sr=RATE,
                                                                      onstart_flag=onstart_flag)

    tempo_list = tempo_making(interval, onSeqout)  # making accompaniment
    if count % 2 == 0:
        old_interval = interval.copy()
        old_pitches = pitches.copy()

        chord_temp, chord_step = chord_recongnize(interval, pitches)

        if chord_step == 0:  # prevent all 0 pitch
            note_list = old_note_list
        else:
            chord_next = chord_predict(chord_record[1], chord_temp, chord_prob_table)
            note_list = note_making(tempo_list, chord_next, chord_step)
            old_note_list = note_list
            if is_offline:
                midi_record(tempo_list, chord_next, midirecord)
    else:
        if old_interval.shape[0] > 0 and interval.shape[0] > 0:
            new_interval = np.concatenate((old_interval, interval), axis=0)
            new_pitches = np.concatenate((old_pitches, pitches), axis=0)
        elif interval.shape[0] == 0:
            new_interval = old_interval
            new_pitches = old_pitches
        else:
            new_interval = interval
            new_pitches = pitches

        chord_temp, chord_step = chord_recongnize(new_interval, new_pitches)
        if chord_step == 0:
            note_list = old_note_list

        else:
            chord_record[0] = chord_record[1]
            chord_record[1] = chord_temp
            chord_next = chord_predict(chord_record[0], chord_record[1], chord_prob_table)
            note_list = note_making(tempo_list, chord_next, chord_step)
            old_note_list = note_list
            if is_offline:
                midi_record(tempo_list, chord_next, midirecord)

    mide_file = create_MIDI(note_list, count=count, volume_para=(volume_cur_para/volume_ori_para))

    end = time.time()
    print(end - start)
    time_cost += end - start
    iter_num+=1
    if count>4:
        max_time.append(end - start)

    print(count)
    if not is_offline:
        while True:
            # print("time:", time.time() - music_start)
            # if not pygame.mixer.music.get_busy() :  #
            if time.time() - music_start> bar_limit:
                SHOOT_SOUND.play()

                pygame.mixer.music.load(mide_file)
                pygame.mixer.music.play()
                print("music_dur: ", time.time() - music_start)

                music_start = time.time()
                # load_flag = False
                break

            else:
                sleep(0.01)

    print(count)
    print(interval, pitches)


if is_offline:
    output_integration("midi_check", "wav_check")
    print(f"iter: {iter_num}, total time: {time_cost}, avg. time: {time_cost/iter_num}")
    max_time = np.sort(np.array(max_time))
    print("max time:", max_time[-10:])