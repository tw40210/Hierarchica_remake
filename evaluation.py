import numpy as np
import librosa
from preprocess import output_feature_extraction_nosave
import hparam
from utils import signal_sampletest_stream, rawout2interval_picth, get_Resnet
import torch


def soloCliptest(file_path,model, RATE=16000):
    buffer = np.zeros((hparam.FEAT_freqbin_num * hparam.FEAT_channel, hparam.FEAT_pastpad + hparam.FEAT_futurepad))
    wavform_buffer = np.zeros((int(RATE * hparam.timestep * hparam.FEAT_futurepad)))
    data_float, sr = librosa.load(file_path, sr=RATE)
    SN_SIN_ZN = output_feature_extraction_nosave(data_float, window_size=[768, 372, 186])
    record, buffer = signal_sampletest_stream(SN_SIN_ZN,past_buffer=buffer, model=model, channel=hparam.FEAT_channel)
    # est_intervals, _, _, _, _, _, onstart_flag = Smooth_sdt6(record, realtime=True, onstart_flag= onstart_flag)

    padding_data_float = data_float[int(-RATE*hparam.timestep*(hparam.FEAT_pastpad)):]
    data_float = np.concatenate((wavform_buffer, data_float[:int(-RATE*hparam.timestep*(hparam.FEAT_pastpad))]), axis=0) # adjust wav signal to match label
    wavform_buffer = padding_data_float
    librosa.output.write_wav(f"wav_check/solotest.wav", data_float, sr=RATE)
    interval, pitches, onstart_flag, onSeqout = rawout2interval_picth(record, data_float, sr=RATE, onstart_flag=False)
    return interval, pitches

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_Resnet(hparam.FEAT_channel).to(device)
    model.load_state_dict(torch.load("standard_checkpoint/960_1030perform077.pth"))
    print("load OK")
    interval, pitches = soloCliptest(file_path="5.wav", model= model)

    print("interval: ", interval)
    print("fin")