import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
import librosa
import argparse
import hparam
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from typing import Dict, List
import os
import torch
from tqdm import tqdm
from model import ResNet, BasicBlock, get_BCE_loss, ResNet_simple
import torch.nn as nn
from tensorboardX import SummaryWriter
import shutil
import mir_eval
import pathlib
import random
import pyworld as pw


# plt.switch_backend('agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class Logger(object):
    def __init__(self, runs_dir):
        self.runs_dir = runs_dir

    def get_runsdir(self):
        ori_dir = os.listdir(self.runs_dir)
        return ori_dir

    def get_new_runsdir(self, ori_runlist):
        new_dirs = os.listdir(self.runs_dir)
        for new_dir in new_dirs:
            if new_dir not in ori_runlist:
                return os.path.join(self.runs_dir, new_dir)
        print("dir is  not found.")

    def save_modelbackup(self, model, tar_dir):
        torch.save(model.state_dict(), os.path.join(tar_dir, "model.pth"))

    def save_codebackup(self, code_dir, tar_dir, index=''):
        shutil.copy(code_dir, os.path.join(tar_dir, f"codebackup{index}.py"))


def silence_label(num_label, num_class):
    assert num_class == 6
    padding = np.repeat(np.array([1, 0, 0, 1, 0, 1]), [num_label, 1])
    return padding


def _sub_expand_onoff_label(label_note, idx, cls):
    for sub_idx in [1, -1]:

        if label_note[idx + sub_idx][cls] != 1 and label_note[idx + sub_idx][0] != 1:
            label_note[idx + sub_idx][cls] = 1
            label_note[idx + sub_idx][cls + 1] = 0

    return label_note


def expand_onoff_label(label_note):
    record_on = []
    record_off = []
    for idx, note in enumerate(label_note):
        if note[2] == 1:
            record_on.append(idx)
        if note[4] == 1:
            record_off.append(idx)

    for idx in record_on:
        if (idx > 1 and idx < len(label_note) - 3):
            label_note = _sub_expand_onoff_label(label_note, idx, 2)

    for idx in record_off:
        if (idx > 1 and idx < len(label_note) - 3):
            label_note = _sub_expand_onoff_label(label_note, idx, 4)

    return label_note


def find_first_bellow_thres(aSeq, on_insert):
    activate = False
    transit_flag=False
    first_bellow_frame = 0
    for i in range(len(aSeq)):
        if aSeq[i] > 0.5:
            activate = True
        if activate and aSeq[i] < 0.5:
            first_bellow_frame = i
            break

    if first_bellow_frame==0 and on_insert:
        aSeq = np.array(aSeq)
        if aSeq.mean()< 0.5:
            transit_flag = True

    elif first_bellow_frame==0 and not on_insert:
        aSeq = np.array(aSeq)
        if aSeq.mean()> 0.5:
            transit_flag = True



    return first_bellow_frame, transit_flag

def power_filter(seq, idx ,filter_size=3):
    power=0
    count=0
    start_idx = idx - ((filter_size-1)//2)
    if start_idx<0:
        start_idx=0

    for seq_idx in range(filter_size):
        if start_idx+seq_idx<len(seq)-1:
            power+= seq[start_idx+seq_idx]
            count+=1

    power /= count
    return power

def Smooth_sdt6(predict_sdt, threshold=0.20, realtime=False, onstart_flag=False, onSeqout=False):
    # predict shape: (time step, 3)

    Filter = np.ndarray(shape=(5,), dtype=float, buffer=np.array([0.25, 0.5, 1.0, 0.5, 0.25]))
    # Filter = np.ndarray(shape=(5,), dtype=float, buffer=np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    sSeq = []
    dSeq = []
    onSeq = []
    offSeq = []

    if onSeqout:
        onSeqout_Seq=[]

    for num in range(predict_sdt.shape[0]):
        if num > 1 and num < predict_sdt.shape[0] - 2:
            sSeq.append(predict_sdt[num][0])
            dSeq.append(predict_sdt[num][1])
            onSeq.append(np.dot(predict_sdt[num - 2:num + 3, 2], Filter) / 2.5)
            offSeq.append(np.dot(predict_sdt[num - 2:num + 3, 4], Filter) / 2.5)

        else:
            sSeq.append(predict_sdt[num][0])
            dSeq.append(predict_sdt[num][1])
            onSeq.append(predict_sdt[num][2])
            offSeq.append(predict_sdt[num][4])


    ##############################
    # Peak strategy
    ##############################

    # find peak of transition
    # peak time = frame*0.02+0.01
    onpeaks = []
    if onstart_flag:
        onpeaks.append(0)
    else:
        if onSeq[0] > onSeq[1] and onSeq[0] > onSeq[2] and onSeq[0] > threshold:
            onpeaks.append(0)
        if onSeq[1] > onSeq[0] and onSeq[1] > onSeq[2] and onSeq[1] > onSeq[3] and onSeq[1] > threshold:
            onpeaks.append(1)

    for num in range(len(onSeq)):
        if num > 1 and num < len(onSeq) - 2:
            if onSeq[num] > onSeq[num - 1] and onSeq[num] > onSeq[num - 2] and onSeq[num] > onSeq[num + 1] and onSeq[
                num] > onSeq[num + 2] and onSeq[num] > threshold:
                onpeaks.append(num)

    if onSeq[-1] > onSeq[-2] and onSeq[-1] > onSeq[-3] and onSeq[-1] > threshold:
        onpeaks.append(len(onSeq) - 1)
    if onSeq[-2] > onSeq[-1] and onSeq[-2] > onSeq[-3] and onSeq[-2] > onSeq[-4] and onSeq[-2] > threshold:
        onpeaks.append(len(onSeq) - 2)

    offpeaks = []
    if offSeq[0] > offSeq[1] and offSeq[0] > offSeq[2] and offSeq[0] > threshold:
        offpeaks.append(0)
    if offSeq[1] > offSeq[0] and offSeq[1] > offSeq[2] and offSeq[1] > offSeq[3] and offSeq[1] > threshold:
        offpeaks.append(1)
    for num in range(len(offSeq)):
        if num > 1 and num < len(offSeq) - 2:
            if offSeq[num] > offSeq[num - 1] and offSeq[num] > offSeq[num - 2] and offSeq[num] > offSeq[num + 1] and \
                    offSeq[num] > offSeq[num + 2] and offSeq[num] > threshold:
                offpeaks.append(num)

    if offSeq[-1] > offSeq[-2] and offSeq[-1] > offSeq[-3] and offSeq[-1] > threshold:
        offpeaks.append(len(offSeq) - 1)
    if offSeq[-2] > offSeq[-1] and offSeq[-2] > offSeq[-3] and offSeq[-2] > offSeq[-4] and offSeq[-2] > threshold:
        offpeaks.append(len(offSeq) - 2)

    # determine onset/offset by silence, duration
    # intervalSD = [0,1,0,1,...], 0:silence, 1:duration
    if len(onpeaks) == 0 or len(offpeaks) == 0:

        if realtime:
            return np.array([]), 0, onstart_flag, onSeqout_Seq
        else:
            return np.array([]), 0

    Tpeaks = onpeaks + offpeaks
    Tpeaks.sort()

    on_off_belong = []
    skip_flag=False
    for idx in range(len(Tpeaks)) :
        if skip_flag:
            skip_flag=False
            continue

        if Tpeaks[idx] in onpeaks and Tpeaks[idx] in offpeaks:
            on_off_belong.append(0)
            on_off_belong.append(1)
            skip_flag=True

        elif Tpeaks[idx] in onpeaks:
            on_off_belong.append(1)
        elif Tpeaks[idx] in offpeaks:
            on_off_belong.append(0)

    assert len(on_off_belong)==len(Tpeaks)
    print(len(on_off_belong), len(Tpeaks))


    intervalSD = []
    current_sd = 0 if sum(sSeq[0:Tpeaks[0]]) > sum(dSeq[0:Tpeaks[0]]) else 1
    intervalSD.append(current_sd)

    for i in range(len(Tpeaks) - 1):
        current_sd = 0 if sum(sSeq[Tpeaks[i]:Tpeaks[i + 1]]) > sum(dSeq[Tpeaks[i]:Tpeaks[i + 1]]) else 1
        intervalSD.append(current_sd)
    current_sd = 0 if sum(sSeq[Tpeaks[-1]:]) > sum(dSeq[Tpeaks[-1]:]) else 1
    intervalSD.append(current_sd)


    MissingT = 0
    AddingT = 0
    est_intervals = []
    t_idx = 0
    while t_idx < len(Tpeaks) - 1:

        if t_idx == 0 and on_off_belong[t_idx] != 1:
            if intervalSD[0] == 1 and intervalSD[1] == 0:
                onset_inserted, transit_flag = find_first_bellow_thres(sSeq[0:Tpeaks[0]], on_insert=True)
                if onset_inserted != Tpeaks[0] and Tpeaks[0] > onset_inserted + 1:
                    est_intervals.append([0.02 * onset_inserted + 0.01, 0.02 * Tpeaks[0] + 0.01])
                    if onSeqout:
                        onSeqout_Seq.append(power_filter(sSeq, onset_inserted))
                    AddingT += 1
                elif transit_flag:
                    est_intervals.append([0.02 * onset_inserted + 0.01, 0.02 * Tpeaks[t_idx] + 0.01])
                    if onSeqout:
                        onSeqout_Seq.append(power_filter(sSeq, onset_inserted))
                    AddingT += 1
                else:
                    MissingT += 1
            t_idx += 1
            continue
        elif t_idx == 0 and onstart_flag and on_off_belong[t_idx + 1] != 0:
            if intervalSD[0] == 1 and intervalSD[1] == 0:
                onset_inserted, transit_flag = find_first_bellow_thres(sSeq[0:Tpeaks[t_idx + 1]], on_insert=True)
                if onset_inserted != Tpeaks[0] and Tpeaks[0] > onset_inserted + 1:
                    est_intervals.append([0.02 * onset_inserted + 0.01, 0.02 * Tpeaks[0] + 0.01])
                    if onSeqout:
                        onSeqout_Seq.append(power_filter(sSeq, onset_inserted))
                    AddingT += 1
                elif transit_flag:
                    est_intervals.append([0.02 * onset_inserted + 0.01, 0.02 * Tpeaks[t_idx] + 0.01])
                    if onSeqout:
                        onSeqout_Seq.append(power_filter(sSeq, onset_inserted))
                    AddingT += 1
                else:
                    MissingT += 1
            t_idx += 1
            continue

        if on_off_belong[t_idx] == 1 and on_off_belong[t_idx + 1] == 0:
            if Tpeaks[t_idx] == Tpeaks[t_idx + 1]:
                t_idx += 1
                continue
            if Tpeaks[t_idx + 1] > Tpeaks[t_idx] + 1:
                est_intervals.append([0.02 * Tpeaks[t_idx] + 0.01, 0.02 * Tpeaks[t_idx + 1] + 0.01])
                if onSeqout:
                    onSeqout_Seq.append(power_filter(sSeq, Tpeaks[t_idx]))
            assert (Tpeaks[t_idx] < Tpeaks[t_idx + 1])
            t_idx += 2
            if t_idx > len(Tpeaks) - 2:
                break
        elif on_off_belong[t_idx] == 1 and on_off_belong[t_idx + 1] == 1:
            offset_inserted, transit_flag = find_first_bellow_thres(dSeq[Tpeaks[t_idx]:Tpeaks[t_idx + 1]],  on_insert=False)
            offset_inserted += Tpeaks[t_idx]
            if offset_inserted != Tpeaks[t_idx] and offset_inserted > Tpeaks[t_idx] + 1:
                est_intervals.append([0.02 * Tpeaks[t_idx] + 0.01, 0.02 * offset_inserted + 0.01])
                if onSeqout:
                    onSeqout_Seq.append(power_filter(sSeq, Tpeaks[t_idx]))
                AddingT += 1
                assert (Tpeaks[t_idx] < offset_inserted)

            elif transit_flag:
                est_intervals.append([0.02 * Tpeaks[t_idx] + 0.01, 0.02 * Tpeaks[t_idx+1] + 0.01])
                if onSeqout:
                    onSeqout_Seq.append(power_filter(sSeq, Tpeaks[t_idx]))
                AddingT += 1
            else:
                MissingT += 1

            t_idx += 1
        elif on_off_belong[t_idx] == 0:
            if intervalSD[t_idx] == 1 and intervalSD[t_idx + 1] == 0:
                onset_inserted, transit_flag = find_first_bellow_thres(sSeq[Tpeaks[t_idx - 1]:Tpeaks[t_idx]], on_insert=True)
                onset_inserted += Tpeaks[t_idx - 1]
                if onset_inserted > Tpeaks[t_idx - 1]-1 and Tpeaks[t_idx] > onset_inserted + 1:
                    est_intervals.append([0.02 * onset_inserted + 0.01, 0.02 * Tpeaks[t_idx] + 0.01])
                    if onSeqout:
                        onSeqout_Seq.append(power_filter(sSeq, onset_inserted))
                    AddingT += 1
                    assert (onset_inserted < Tpeaks[t_idx])
                elif transit_flag:
                    est_intervals.append([0.02 * onset_inserted + 0.01, 0.02 * Tpeaks[t_idx] + 0.01])
                    if onSeqout:
                        onSeqout_Seq.append(power_filter(sSeq, onset_inserted))
                    AddingT += 1
                else:
                    MissingT += 1
            elif intervalSD[t_idx] == 1 and intervalSD[t_idx + 1] == 1 and on_off_belong[t_idx - 1] == 0:
                onset_inserted,transit_flag = find_first_bellow_thres(sSeq[Tpeaks[t_idx - 1]:Tpeaks[t_idx]], on_insert=True)
                onset_inserted += Tpeaks[t_idx - 1]
                if onset_inserted != Tpeaks[t_idx - 1] and Tpeaks[t_idx] > onset_inserted + 1:
                    est_intervals.append([0.02 * onset_inserted + 0.01, 0.02 * Tpeaks[t_idx] + 0.01])
                    if onSeqout:
                        onSeqout_Seq.append(power_filter(sSeq, onset_inserted))
                    AddingT += 1
                    assert (onset_inserted < Tpeaks[t_idx])

                elif transit_flag:
                    est_intervals.append([0.02 * onset_inserted + 0.01, 0.02 * Tpeaks[t_idx] + 0.01])
                    if onSeqout:
                        onSeqout_Seq.append(power_filter(sSeq, onset_inserted))
                    AddingT += 1

                else:
                    MissingT += 1

            t_idx += 1

    if realtime:
        if len(Tpeaks) > 1:
            if on_off_belong[-1] == 1 and intervalSD[-1] == 1:
                est_intervals.append([0.02 * Tpeaks[-1], (60/hparam.song_bpm)*2])
                if onSeqout:
                    onSeqout_Seq.append( power_filter(sSeq,Tpeaks[-1]))
                onstart_flag = True

            else:
                onstart_flag = False

    # print("Missing ratio: ", MissingT/len(est_intervals))
    print("Conflict ratio: ", MissingT / (len(Tpeaks) + AddingT))

    # Modify 1


    if realtime:
        return np.array(est_intervals, dtype=float), MissingT / (
                    len(Tpeaks) + AddingT), onstart_flag, np.array(onSeqout_Seq)

    return np.array(est_intervals, dtype=float), MissingT / (
            len(Tpeaks) + AddingT)


def get_Resnet(channel=9, is_simplified=False):
    if is_simplified:
        model = ResNet_simple(BasicBlock, [2, 2, 2, 2])
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2])
    num_fout = model.conv1.out_channels
    model.conv1 = nn.Conv2d(channel, num_fout, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                            bias=False)
    model.fc = nn.Linear(model.fc.in_features, 6)
    model.avgpool = nn.AvgPool2d(kernel_size=(6, 1), stride=1, padding=0)

    return model


def testset_evaluation(path, f_path, model=None, writer_in=None, timestep=None, is_plot=False, channel=9):
    if not model:
        model = get_Resnet(channel=channel)
    if not writer_in:
        writer = SummaryWriter(comment="test_seperated1122")
    else:
        writer = writer_in

    if not timestep:
        timestep = 0

    wav_files = [os.path.join(path, file) for file in os.listdir(path) if '.wav' in file]
    labels = [os.path.join(path, label) for label in os.listdir(path) if '.notes.' in label]
    features = [os.path.join(f_path, features) for features in os.listdir(f_path) if '_FEAT' in features]

    sum_on_F1 = 0
    sum_off_F1 = 0
    sum_note_F = 0

    model.eval()
    count = 0
    print("testing on testset for on/off_F1\n")
    with torch.no_grad():
        for index in range(len(features)):
            # if count<3:
            #     count += 1
            #     continue

            # if count > 4:  # shorten test time by sampling
            #     break
            record = []
            features_full = np.load(features[index])
            label_path = str(pathlib.Path(labels[index]).parent / (
                    pathlib.Path(features[index]).stem.split('.')[0] + ".notes.Corrected"))
            wav_path = str(pathlib.Path(labels[index]).parent / (
                    pathlib.Path(features[index]).stem.split('.')[0] + ".wav"))
            label_note = read_notefile(label_path)
            gt_label_sec_on, gt_label_sec_off = note2onoff_sec(label_note)

            label_note, label_pitch = note2timestep(label_note)
            label_note = np.array(label_note)
            label_pitch = np.array(label_pitch)

            # cut muted tail from feature
            features_full = features_full[:, :label_note.shape[0]]
            # pad 9 zero steps in both head and tail
            zero_pad = np.zeros((features_full.shape[0], 9))
            features_full = np.concatenate((zero_pad, features_full), axis=1)
            features_full = np.concatenate((features_full, zero_pad), axis=1)
            features_full = abs(features_full)
            features_full = np.power(features_full / features_full.max(), hparam.gamma_mu)  # normalize &gamma compression

            for test_step in range(features_full.shape[1] - 18):
                curr_clip = features_full[:, test_step:test_step + 19]
                curr_clip = torch.from_numpy(curr_clip)
                curr_clip = curr_clip.view(channel, 174, -1).float()
                curr_clip = curr_clip.unsqueeze(0)
                curr_clip = curr_clip.to(device)
                model = model.to(device)
                out_label = model(curr_clip)
                out_label = out_label.squeeze(0).squeeze(0).cpu().detach().numpy()

                record.append(out_label)

            record = np.array(record)
            print(features[index])
            est_intervals, _, = Smooth_sdt6(record)
            if len(est_intervals)!=0:

                est_smooth_label_sec_on = []
                est_smooth_label_sec_off = []
                for interval in est_intervals:
                    est_smooth_label_sec_on.append(interval[0])
                    est_smooth_label_sec_off.append(interval[1])
                est_smooth_label_sec_on = np.array(est_smooth_label_sec_on)
                est_smooth_label_sec_off = np.array(est_smooth_label_sec_off)

                # ===== pitch trace
                est_pitch = interval2pitch_in_note(est_intervals, wavfile=wav_path)

                gt_interval_array = onoffarray2interval(gt_label_sec_on, gt_label_sec_off)
                gt_notes = gt_pitch_in_note(gt_interval_array, label_pitch)

                on_F, on_P, on_R = mir_eval.onset.f_measure(gt_label_sec_on, est_smooth_label_sec_on)
                off_F, off_P, off_R = mir_eval.onset.f_measure(gt_label_sec_off, est_smooth_label_sec_off)
                notes_evaluation = mir_eval.transcription.evaluate(gt_interval_array, gt_notes, est_intervals, est_pitch)

                note_F = notes_evaluation["F-measure"]

                if is_plot:
                    plot_note(gt_interval_array, gt_notes, est_intervals, est_pitch)

                sum_on_F1 += on_F
                sum_off_F1 += off_F
                sum_note_F += note_F
                count += 1
                print(f"smooth_on_F1: {on_F}, smooth_off_F1: {off_F}, note_F: {note_F}")

    writer.add_scalar(f'scalar/onoff/on_F1', sum_on_F1 / count, timestep)
    writer.add_scalar(f'scalar/onoff/off_F1', sum_off_F1 / count, timestep)
    writer.add_scalar(f'scalar/onoff/note_F', sum_note_F / count, timestep)
    print(f"ALL: smooth_on_F1: {sum_on_F1 / count}, smooth_off_F1: {sum_off_F1 / count}, note_F: {sum_note_F / count}")

    #
    # writer.add_scalars(f"scalar\\onoff_F1", {'on_F1': sum_on_F1 / count, 'off_F1': sum_off_F1 / count}, timestep)


def rawout2interval_picth(record, signal, sr, onstart_flag):
    est_intervals, _, onstart_flag, onSeqout = Smooth_sdt6(record, realtime=True, onstart_flag=onstart_flag, onSeqout=True)

    est_pitch = interval2pitch_in_note(est_intervals, signal=signal, signal_only=True, sr=sr)
    return est_intervals, est_pitch, onstart_flag, onSeqout


def signal_sampletest_stream(input_x, past_buffer, model=None, writer_in=None, timestep=None,
                             channel=hparam.FEAT_channel):
    if not model:
        model = get_Resnet()
    if not writer_in:
        writer = SummaryWriter()
    else:
        writer = writer_in

    if not timestep:
        timestep = 0

    model = model.to(device)
    model.eval()
    count = 0

    record = []
    future_buffersize = hparam.FEAT_futurepad
    past_buffersize = hparam.FEAT_pastpad

    features_full = input_x

    # cut muted tail from feature
    # features_full = features_full[:, :label_note.shape[0]]
    # pad 9 zero steps in both head and tail
    assert past_buffer.shape == (features_full.shape[0], past_buffersize + future_buffersize)
    padding = past_buffer
    features_full = np.concatenate((padding, features_full), axis=1)  # only do past padding
    features_full = abs(features_full)
    features_full = np.power(features_full / features_full.max(), hparam.gamma_mu)  # normalize &gamma compression

    for test_step in range(features_full.shape[1] - 18):
        curr_clip = features_full[:, test_step:test_step + 19]
        curr_clip = torch.from_numpy(curr_clip)
        curr_clip = curr_clip.view(channel, 174, -1).float()
        curr_clip = curr_clip.unsqueeze(0)
        curr_clip = curr_clip.to(device)

        out_label = model(curr_clip)
        out_label = out_label.squeeze(0).squeeze(0).cpu().detach().numpy()

        record.append(out_label)
        count += 1

    record = np.array(record)
    furture_buffer = features_full[:, - (past_buffersize + future_buffersize):]

    return record, furture_buffer


def whole_song_sampletest(path, f_path, model=None, writer_in=None, timestep=None, channel=9):
    if not model:
        model = get_Resnet(channel=channel)
    if not writer_in:
        writer = SummaryWriter()
    else:
        writer = writer_in

    if not timestep:
        timestep = 0

    wav_files = [os.path.join(path, file) for file in os.listdir(path) if '.wav' in file]
    labels = [os.path.join(path, label) for label in os.listdir(path) if '.notes.' in label]
    features = [os.path.join(f_path, features) for features in os.listdir(f_path) if '_FEAT' in features]

    model.eval()
    count = 0
    with torch.no_grad():
        for index in range(len(features)):
            record = []

            features_full = np.load(features[index])

            a = features[index]
            label_path = str(pathlib.Path(labels[index]).parent / (
                    pathlib.Path(features[index]).stem.split('.')[0] + ".notes.Corrected"))

            label_note = read_notefile(label_path)
            label_note, label_pitch = note2timestep(label_note)
            label_note = np.array(label_note)
            label_pitch = np.array(label_pitch)

            # cut muted tail from feature
            features_full = features_full[:, :label_note.shape[0]]
            # pad 9 zero steps in both head and tail
            zero_pad = np.zeros((features_full.shape[0], 9))
            features_full = np.concatenate((zero_pad, features_full), axis=1)
            features_full = np.concatenate((features_full, zero_pad), axis=1)
            features_full = abs(features_full)
            features_full = np.power(features_full / features_full.max(), hparam.gamma_mu)  # normalize &gamma compression

            for test_step in range(features_full.shape[1] - 18):
                curr_clip = features_full[:, test_step:test_step + 19]
                curr_clip = torch.from_numpy(curr_clip)
                curr_clip = curr_clip.contiguous().view(channel, 174, -1).float()
                curr_clip = curr_clip.unsqueeze(0)
                curr_clip = curr_clip.to(device)
                model = model.to(device)
                out_label = model(curr_clip)
                out_label = out_label.squeeze(0).squeeze(0).cpu().detach().numpy()

                record.append(out_label)
                count += 1

        record = np.array(record)

        plt.figure(figsize=(7, 12))
        plt.subplots_adjust(wspace=0, hspace=1)

        for la_idx in range(record.shape[1]):
            plt.subplot(record.shape[1], 1, la_idx + 1)
            plt.title(f"{la_idx}")
            plt.ylim(0, 1)
            plt.plot(record[:hparam.whole_song_max_len, la_idx])

        fig = plt.gcf()
        writer.add_figure(f"figurs\\{index}", fig, timestep)

        # ====GT
        if timestep < hparam.step_to_save + 1:
            plt.figure(figsize=(7, 12))
            plt.subplots_adjust(wspace=0, hspace=1)

            for la_idx in range(label_note.shape[1] // 2):
                la_idx *= 2
                plt.subplot(label_note.shape[1] // 2, 1, la_idx // 2 + 1)
                plt.title(f"{la_idx}_gt")
                plt.ylim(-0.2, 1.2)
                plt.plot(label_note[:hparam.whole_song_max_len, la_idx])

            fig = plt.gcf()
            writer.add_figure(f"figurs\\{index}_gt", fig, timestep)

        # plt.show()
    if not writer_in:
        writer.close()


def get_accuracy(est_label, ref_label):
    correct = 0
    total = ref_label.shape[0] * ref_label.shape[1]
    est_label = output2label(est_label)
    ref_label = ref_label.int()

    for batch_idx in range(ref_label.shape[0]):
        for frame_idx in range(ref_label.shape[1]):
            if torch.equal(est_label[batch_idx][frame_idx], ref_label[batch_idx][frame_idx]):
                correct += 1

    return correct / total


def read_sheetlabel(path):
    gt_sheetlabel=[]
    downbeats=[]

    with open(path, 'r') as gt_file:
        tempo_count=0
        tmp_list=[]
        tmp_beatlist=[]
        for line in gt_file.readlines():
            data= line.split(',')

            if data[3].strip()!="0":
                tmp_list.append(data[3].strip())
                tmp_beatlist.append(tempo_count)
            if data[1]=="D":
                downbeats.append(tempo_count)
            tempo_count += int(data[0])

            if tempo_count>=16:
                gt_sheetlabel.append([downbeats, data[2].strip(),tmp_list,tmp_beatlist])
                tempo_count = 0
                downbeats = []
                tmp_list = []
                tmp_beatlist = []

    return gt_sheetlabel






def read_notefile(path, limit_len=None):
    notes = []
    with open(path, 'r') as txt:
        lines = txt.readlines()
        lines = lines[1:]
        for line in lines:
            note = list(map(float, line.split(', ')))
            notes.append(note)

    return notes


def output2label(est_output, is_batch=True, is_nparray=False):
    if is_batch:
        for batch_idx in range(est_output.shape[0]):
            for frame_idx in range(est_output.shape[1]):
                norm_sa = est_output[batch_idx][frame_idx][0] + est_output[batch_idx][frame_idx][1]
                norm_on = est_output[batch_idx][frame_idx][2] + est_output[batch_idx][frame_idx][
                    3]  # make sure the sum of on and Xon =1
                norm_off = est_output[batch_idx][frame_idx][4] + est_output[batch_idx][frame_idx][5]
                est_output[batch_idx][frame_idx][0] /= norm_sa
                est_output[batch_idx][frame_idx][1] /= norm_sa
                est_output[batch_idx][frame_idx][2] /= norm_on
                est_output[batch_idx][frame_idx][3] /= norm_on
                est_output[batch_idx][frame_idx][4] /= norm_off
                est_output[batch_idx][frame_idx][5] /= norm_off
    else:
        for frame_idx in range(est_output.shape[0]):
            norm_sa = est_output[frame_idx][0] + est_output[frame_idx][1]
            norm_on = est_output[frame_idx][2] + est_output[frame_idx][
                3]  # make sure the sum of on and Xon =1
            norm_off = est_output[frame_idx][4] + est_output[frame_idx][5]
            est_output[frame_idx][0] /= norm_sa
            est_output[frame_idx][1] /= norm_sa
            est_output[frame_idx][2] /= norm_on
            est_output[frame_idx][3] /= norm_on
            est_output[frame_idx][4] /= norm_off
            est_output[frame_idx][5] /= norm_off

    if is_nparray:
        est_label = np.where(est_output > hparam.label_threshold, 1, 0)
    else:
        est_label = (est_output > hparam.label_threshold).int()
    return est_label


def timestep2second(labels):
    onset_list = []
    offset_list = []

    for timestep, label in enumerate(labels):
        if (label[2] == 1):
            onset_list.append(timestep * hparam.timestep + hparam.timestep * 0.5)

    for timestep, label in enumerate(labels):
        if (label[4] == 1):
            offset_list.append(timestep * hparam.timestep + hparam.timestep * 0.5)

    return np.array(onset_list), np.array(offset_list)


def note2onoff_sec(notes: List):
    onset_list = []
    offset_list = []

    for timestep, label in enumerate(notes):
        onset_list.append(label[0])
        offset_list.append(label[0] + label[1])

    return np.array(onset_list), np.array(offset_list)


def note2timestep(notes: List):
    timestep = []
    pitch = []
    tail = 0
    end_tail = 0
    for idx, note in enumerate(notes):
        status = [1, 0, 0, 1, 0, 1]  # S, A, O, -O, X, -X
        while (len(timestep) < note[0] // 0.02):
            timestep.append(status)
            pitch.append(0)

        if idx > 0:
            if note[0] - end_tail < 1e-4:
                timestep[-1] = [0, 1, 1, 0, 1, 0]
                pitch[-1] = (note[2])
            else:
                status = [0, 1, 1, 0, 0, 1]
                timestep.append(status)
                pitch.append(note[2])
        else:
            status = [0, 1, 1, 0, 0, 1]
            timestep.append(status)
            pitch.append(note[2])

        tail = len(timestep) * 0.02
        end_tail = (note[0] + note[1]) // 0.02 * 0.02 + 0.02
        status = [0, 1, 0, 1, 0, 1]

        for _ in range(int((note[0] + note[1] - tail + 1e-4) // 0.02)):
            timestep.append(status)
            pitch.append(note[2])

        status = [0, 1, 0, 1, 1, 0]
        timestep.append(status)
        pitch.append(note[2])

    return timestep, pitch


def get_pitch_steps():
    pitch_steps = []
    A4 = 440
    midi_id = 69
    start = A4 * np.power(1 / 2, midi_id / 12)
    step = np.power(2, 1 / 12)
    pitch_steps.append(start)
    for i in range(127):
        start = start * step
        pitch_steps.append(start)

    return pitch_steps


def pick_pitch(pitches, pitch_steps):
    pitch_midiid = []

    for pitch in pitches:
        pitch_id = 0
        while (pitch > pitch_steps[pitch_id]):
            pitch_id += 1

        if (abs(pitch_steps[pitch_id] - pitch) > abs(pitch_steps[pitch_id + 1] - pitch)):
            pitch_id = pitch_id + 1

        pitch_midiid.append(pitch_id)

    mid_pitch = np.median(np.array(pitch_midiid))
    return mid_pitch


def sec2sample(sec_array, timeresolution=200):
    sec_array *= timeresolution
    sample_array = np.array(list(map(int, sec_array)))
    return sample_array


def gt_pitch_in_note(intervals, pitch_label, time_resolution=0.02):
    second_length = int(1 / time_resolution)
    gt_pitch = []

    for interval in intervals:
        pitch_interval = pitch_label[int(interval[0] * second_length): int(interval[1] * second_length)]
        gt_pitch.append(np.median(np.array(pitch_interval)))

    return np.array(gt_pitch)


def freq2octal(_f0, pitch_steps):
    octal_f0 = []
    for f0 in _f0:
        if f0 < 6:
            octal_f0.append(0)
            continue
        for idx in range(len(pitch_steps) - 2):
            if pitch_steps[idx + 1] > f0:
                octal_f0.append(((f0 - pitch_steps[idx]) / (pitch_steps[idx + 1] - pitch_steps[idx]) + idx))
                break

    return octal_f0


def smoothPitch(pitch_midi_list):
    pitch_midi_list = np.array(pitch_midi_list)
    std_pitch_step = (pitch_midi_list.sum() / pitch_midi_list.shape[0]) // 12

    pitch_midi_list[0] = pitch_midi_list[0] % 12 + std_pitch_step * 12

    for idx in range(len(pitch_midi_list) - 1):
        if pitch_midi_list[idx + 1] - pitch_midi_list[idx] > 12:
            pitch_midi_list[idx + 1] -= 12
        elif pitch_midi_list[idx + 1] - pitch_midi_list[idx] < -12:
            pitch_midi_list[idx + 1] += 12

    return pitch_midi_list


def interval2pitch_in_note(interval, wavfile=None, signal=None, signal_only=False, sr=44100, second_length=200,
                           is_plot=None):
    pitch_steps = get_pitch_steps()

    pitch_midi_list = []

    if wavfile and not signal_only:
        y, sr = librosa.load(wavfile, sr=sr, dtype="double")
    elif signal_only:
        y = np.array(signal, dtype="double")
    else:
        print("wrong parameters!")
        assert False

    _f0, t = pw.dio(y, sr)

    octal_f0 = np.array(freq2octal(_f0, pitch_steps))

    if is_plot:
        y_major_locator = MultipleLocator(1)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
        plt.ylim(bottom=octal_f0[octal_f0 != 0].min(), top=octal_f0[octal_f0 != 0].max())
        plt.plot(octal_f0)
        plt.show()

    if len(interval) > 0:
        assert interval[-1][1] * second_length <= len(_f0)
    for idx, note in enumerate(interval):
        if _f0.max() < 1:
            pitch_midi_list.append(0)
            continue

        pitches = _f0[int(note[0] * second_length):int(note[1] * second_length)]
        length = len(pitches)
        pitches = pitches[int(length / 2):-int(length / 4)]
        pitches = pitches[pitches != 0]
        pitch_count = 1.2

        # if pitches
        #
        while (pitches.size < 1):
            pitches = _f0[int(note[0] * second_length):int(note[1] * second_length)]
            pitches = pitches[int(length / (2 * pitch_count)):-int(length / (4 * pitch_count))]
            pitches = pitches[pitches != 0]
            pitch_count += 0.1
            if (pitch_count > 2):  # if no pitch take nearest
                note_idx1 = int(note[0] * second_length)
                note_idx2 = int(note[1] * second_length)

                while (note_idx2 < _f0.size - 1):
                    if _f0[note_idx2] > 1:
                        pitches = _f0[note_idx2:note_idx2 + 1]
                        break
                    else:
                        note_idx2 += 1
                while (note_idx1 > 0 and pitches.size < 1):
                    if _f0[note_idx1] > 1:
                        pitches = _f0[note_idx1:note_idx1 + 1]
                        break
                    else:
                        note_idx1 -= 1

        pitch_midi_list.append(pick_pitch(pitches, pitch_steps))

    # smooth pitch to avoid big gap

    if len(pitch_midi_list) > 0:
        pitch_midi_list = smoothPitch(pitch_midi_list)

    return pitch_midi_list


def onoffarray2interval(onset_array, offset_array):
    note_list = []
    tmp_list = []

    oncount = 0
    offcount = 0
    count = 0
    on_id = 0
    off_id = 0
    while (True):

        if off_id > len(offset_array) - 1 and on_id > len(onset_array) - 1:
            break
        elif off_id > len(offset_array) - 1:
            tmp_list.append([onset_array[on_id], 1])
            on_id += 1
            count += 1
            continue
        elif on_id > len(onset_array) - 1:
            tmp_list.append([offset_array[off_id], 0])
            off_id += 1
            count += 1
            continue

        if onset_array[on_id] > offset_array[off_id]:
            tmp_list.append([offset_array[off_id], 0])
            off_id += 1
            count += 1
        else:
            tmp_list.append([onset_array[on_id], 1])
            on_id += 1
            count += 1

    for idx, tmpitem in enumerate(tmp_list):  # remove all off before first on
        if tmpitem[1] == 0:
            offcount += 1
        else:
            break

    for idx, tmpitem in enumerate(tmp_list):  # remove all on after last off

        if tmp_list[-(idx + 1)][1] == 1:
            oncount += 1
        else:
            break

    if oncount == 0:
        tmp_list = tmp_list[offcount:]
    else:
        tmp_list = tmp_list[offcount:-oncount]

    adjust_idx = []
    for id in range(len(tmp_list[:-1])):
        if id == 0:
            if tmp_list[id + 1][1] == 1:
                tmp_list.insert(id + 1, [tmp_list[id + 1][0], 0])
                adjust_idx.append([id + 1, 0])
            continue

        if tmp_list[id][1] == 1 and tmp_list[id + 1][1] == 1:
            adjust_idx.append([id + 1, 0])
        elif tmp_list[id][1] == 0 and tmp_list[id + 1][1] == 0:
            adjust_idx.append([id + 1, -1])

    adj_count = 0
    for item in adjust_idx:
        if item[1] == 0:
            tmp_list.insert(item[0] + adj_count, [tmp_list[item[0] + adj_count][0], 0])
            adj_count += 1
        elif item[1] == -1:
            del tmp_list[item[0] + adj_count]
            adj_count -= 1

    count = 0
    assert len(tmp_list) % 2 == 0
    while (count < len(tmp_list)):
        note_list.append([tmp_list[count][0], tmp_list[count + 1][0]])
        count += 2

    return np.array(note_list)


def plot_note(gt_interval_array, gt_notes, est_intervals, est_pitch):
    gt_notes = np.round(gt_notes, 0)

    new_gt = []
    for idx, interval in enumerate(gt_interval_array):
        new_gt.append([interval[0], gt_notes[idx]])
        new_gt.append([interval[1], gt_notes[idx]])

    new_est = []
    for idx, interval in enumerate(est_intervals):
        new_est.append([interval[0], est_pitch[idx]])
        new_est.append([interval[1], est_pitch[idx]])

    new_gt = np.array(new_gt)
    new_est = np.array(new_est)

    plt.figure(figsize=(6, 4.5), dpi=100)  # 設定圖片尺寸
    plt.xlabel('r (m)', fontsize=16)  # 設定坐標軸標籤
    plt.xticks(fontsize=12)  # 設定坐標軸數字格式
    plt.yticks(fontsize=12)

    plt.plot(new_gt[:, 0], new_gt[:, 1], color='red', linewidth=1, label='GT')
    plt.plot(new_est[:, 0], new_est[:, 1], color='blue', linewidth=1, label='EST')
    plt.show()
    return


if __name__ == '__main__':
    path = 'data/test/EvaluationFramework_ISMIR2014/DATASET'
    f_path = 'data/test/Process_data/FEAT'

    onset = [0.1, 0.256, 0.279, 0.336, 0.469, 0.53]
    offset = [0.01, 0.23, 0.266, 0.267, 0.39, 0.4]

    # model = get_Resnet(channel=hparam.FEAT_channel).to(device)
    # model.load_state_dict(torch.load("checkpoint/1227_748HP.pth"))
    from resnest import resnest50

    model = resnest50(channel=9, num_classes=6).to(device)
    model.load_state_dict(torch.load("checkpoint/3690.pth"))
    model.eval()
    print("load OK")

    testset_evaluation(path, f_path, model=model, timestep=0, channel=hparam.FEAT_channel)
