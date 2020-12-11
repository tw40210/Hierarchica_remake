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
from utils import signal_sampletest_, get_Resnet
import torch

# def callbackln(in_data, frame_count, time_info, status):
#     in_data_int = np.array(struct.unpack(str(2 * CHUNK)+'b', in_data), dtype='b')
#     out_data = output_feature_extraction_nosave(in_data_int)
#     out_data_byte = struct.pack(str(2 * CHUNK)+'b', *out_data)
#     return (out_data_byte, pyaudio.paContinue)


matplotlib.use('TKAgg',warn=False, force=True)
print( "Switched to:",matplotlib.get_backend())
CHUNK=1280
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

stream.start_stream()

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
model = get_Resnet().to(device)
model.load_state_dict(torch.load("checkpoint/5280_augset_1028.pth"))
print("load OK")

while True:

    count+=1
    data = stream.read(CHUNK)
    start = time.time()
    data_int = np.array(struct.unpack(str(2 * CHUNK)+'B', data), dtype='b')[::2] + 0
    data_float = np.array(data_int, dtype=float)/128
    SN_SIN_ZN, Z1, CenFreq1 = output_feature_extraction_nosave(data_float)
    record = signal_sampletest_(SN_SIN_ZN, model=model)
    print(count)
    print(record)
    end = time.time()
    print(end-start)

