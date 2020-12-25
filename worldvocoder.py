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

def callbackln(in_data, frame_count, time_info, status):
    in_data_int = np.array(struct.unpack(str(2 * CHUNK)+'b', in_data), dtype='b')
    out_data = in_data_int
    out_data_byte = struct.pack(str(2 * CHUNK)+'b', *out_data)
    return (out_data_byte, pyaudio.paContinue)


matplotlib.use('TKAgg',warn=False, force=True)
print( "Switched to:",matplotlib.get_backend())
CHUNK=32000
FORMAT = pyaudio.paInt16
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
# while True:
#     if stream.is_active():
#         count+=1
#     if count%1000==0:
#         print(count)



# fig, ax = plt.subplots()
# x = np.arange(0,2*CHUNK,2)
# line, = ax.plot(x,np.random.rand(CHUNK))
# ax.set_ylim(-128,128)
# ax.set_xlim(0,CHUNK)
# fig.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_Resnet(hparam.FEAT_channel).to(device)
# model.load_state_dict(torch.load("checkpoint/5280_augset_1028.pth"))
# print("load OK")
buffer = np.zeros((hparam.FEAT_freqbin_num*hparam.FEAT_channel, hparam.FEAT_pastpad+hparam.FEAT_futurepad))
wavform_buffer = np.zeros((int(RATE*hparam.timestep*hparam.FEAT_futurepad)))

stream.start_stream()

while True:

    count+=1
    data = stream.read(CHUNK)
    start = time.time()
    data_int = np.array(struct.unpack(str(2 * CHUNK)+'B', data), dtype='b')[::2] + 0
    data_float = np.array(data_int, dtype=float)/128

    SN_SIN_ZN, Z1, CenFreq1 = output_feature_extraction_nosave(data_float)
    record, buffer = signal_sampletest_stream(SN_SIN_ZN,past_buffer=buffer, model=model, channel=hparam.FEAT_channel)
    est_intervals, _, _, _, _, _ = Smooth_sdt6(record)
    a = int(RATE*hparam.timestep*(hparam.FEAT_pastpad + hparam.FEAT_futurepad))
    data_float = np.concatenate((wavform_buffer, data_float[:int(-RATE*hparam.timestep*(hparam.FEAT_pastpad))]), axis=0) # adjust wav signal to match label
    wavform_buffer = data_float[int(-RATE * hparam.timestep * hparam.FEAT_futurepad):]

    interval, pitch = rawout2interval_picth(record, data_float, sr=RATE)
    print(count)
    print(interval, pitch)
    end = time.time()
    print(end-start)

