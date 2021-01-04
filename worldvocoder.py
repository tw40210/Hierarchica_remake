import pyaudio
import wave
import time
import sys
import struct
import matplotlib.pyplot as plt
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

global mide_file
global play_lock


def play_midi():
    SHOOT_SOUND = pygame.mixer.Sound('data/bass.wav')
    SHOOT_SOUND.set_volume(0.05)
    global play_lock
    while True:
        if play_lock:
            pygame.mixer.music.load(mide_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                sleep(0.01)
            SHOOT_SOUND.play()
            play_lock = False
            print("Done!")
        else:
            sleep(0.02)

def convert_seconds_to_quarter(time_in_sec, bpm):
    quarter_per_second = (bpm/60)
    time_in_quarter = time_in_sec * quarter_per_second
    return time_in_quarter

def create_MIDI(interval, pitches):

    memFile = BytesIO()
    MyMIDI = MIDIFile(1)
    track = 0
    time = 0
    channel = 0
    pitch = 60
    duration = 1
    volume = 100
    bpm = 600
    MyMIDI.addTrackName(track, time, "Sample Track")
    MyMIDI.addTempo(track, time, bpm)


    for idx, note_duration in enumerate(interval):
        time = convert_seconds_to_quarter(note_duration[0] ,bpm)
        duration = convert_seconds_to_quarter(note_duration[1]- note_duration[0] ,bpm)
        pitch = int(pitches[idx])
        MyMIDI.addNote(track, channel, pitch, time, duration, volume)

    duration = convert_seconds_to_quarter(2.5, bpm)
    MyMIDI.addNote(track, channel, 30, 0, duration, volume)
    if len(interval)>0:
        time = convert_seconds_to_quarter(interval[-1][1], bpm)
        duration = convert_seconds_to_quarter(2 - interval[-1][1], bpm)
        MyMIDI.addNote(track, channel, 0, time, duration, 0)

    else:
        time=0
        duration= convert_seconds_to_quarter(2, bpm)
        MyMIDI.addNote(track, channel, 0, time, duration, 0)


    # WRITE A SCALE
    #
    # MyMIDI.addNote(track, channel, pitch, time, duration, volume)
    # for notestep in [2, 2, 1, 2, 2, 2, 1]:
    #     time += duration
    #     pitch += notestep
    #     MyMIDI.addNote(track, channel, pitch, time, duration, volume)
    MyMIDI.writeFile(memFile)

    memFile.seek(0)

    return memFile


# # memFile.seek(0)  # THIS IS CRITICAL, OTHERWISE YOU GET THAT ERROR!
# pygame.mixer.music.load(memFile)
# pygame.mixer.music.play()
# while pygame.mixer.music.get_busy():
#     sleep(1)
# print "Done!"

# def callbackln(in_data, frame_count, time_info, status):
#     in_data_int = np.array(struct.unpack(str(2 * CHUNK)+'b', in_data), dtype='b')
#     out_data = in_data_int
#     out_data_byte = struct.pack(str(2 * CHUNK)+'b', *out_data)
#     return (out_data_byte, pyaudio.paContinue)


import pygame
import pygame.mixer
from time import sleep

pygame.init()
pygame.mixer.init()

matplotlib.use('TKAgg',warn=False, force=True)
print( "Switched to:",matplotlib.get_backend())
CHUNK=32000
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000



p = pyaudio.PyAudio()

stream = p.open(
    format=FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input =True,
    output = True,
    frames_per_buffer = CHUNK,
    # stream_callback = callbackln
)



count=0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_Resnet(hparam.FEAT_channel).to(device)
model.load_state_dict(torch.load("checkpoint/1227_748HP.pth"))
print("load OK")
buffer = np.zeros((hparam.FEAT_freqbin_num*hparam.FEAT_channel, hparam.FEAT_pastpad+hparam.FEAT_futurepad))
wavform_buffer = np.zeros((int(RATE*hparam.timestep*hparam.FEAT_futurepad)))

input() #pause
stream.start_stream()
t = threading.Thread(target=play_midi)
midi_playing=False

while True:

    count+=1
    data = stream.read(CHUNK)
    start = time.time()
    data_float = np.fromstring(data, 'Float32')

    # data_int = np.array(struct.unpack(str(2 * CHUNK) + 'b', data), dtype='b')
    # data_float = np.array(data_int, dtype=float)/128



    SN_SIN_ZN, Z1, CenFreq1 = output_feature_extraction_nosave(data_float)
    record, buffer = signal_sampletest_stream(SN_SIN_ZN,past_buffer=buffer, model=model, channel=hparam.FEAT_channel)
    est_intervals, _, _, _, _, _ = Smooth_sdt6(record)

    data_float = np.concatenate((wavform_buffer, data_float[:int(-RATE*hparam.timestep*(hparam.FEAT_pastpad))]), axis=0) # adjust wav signal to match label
    wavform_buffer = data_float[int(-RATE * hparam.timestep * hparam.FEAT_futurepad):]
    librosa.output.write_wav(f"wav_check/{count}.wav", data_float, sr=RATE)
    interval, pitches = rawout2interval_picth(record, data_float, sr=RATE)


    mide_file = create_MIDI(interval, pitches)

    print(count)



    if midi_playing:
        # t.join()
        print(play_lock)
        while play_lock:
            sleep(0.01)
        play_lock = True

    else:
        play_lock = True
        t.start()


    midi_playing =True


    print(count)
    print(interval, pitches)
    end = time.time()
    print(end-start)

