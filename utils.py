import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
import librosa
import argparse
import hparam
import matplotlib.pyplot as plt
from typing import Dict, List
import os
import torch
from tqdm import tqdm
from model import ResNet, BasicBlock, get_BCE_loss
import torch.nn as nn
from tensorboardX import SummaryWriter
import shutil
import mir_eval
import pathlib

# plt.switch_backend('agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    def save_codebackup(self, code_dir, tar_dir):
        shutil.copy(code_dir, os.path.join(tar_dir, "codebackup.py"))


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

            # label_note[idx][2]=1
            # label_note[idx-1][2]=1
            # label_note[idx-2][2]=1
            # label_note[idx+1][2]=1
            # label_note[idx+2][2]=1

    for idx in record_off:
        if (idx > 1 and idx < len(label_note) - 3):
            label_note = _sub_expand_onoff_label(label_note, idx, 4)

            # label_note[idx][4]=1
            # label_note[idx-1][4]=1
            # label_note[idx-2][4]=1
            # label_note[idx+1][4]=1
            # label_note[idx+2][4]=1

    return label_note


def find_first_bellow_thres(aSeq):
    activate = False
    first_bellow_frame = 0
    for i in range(len(aSeq)):
        if aSeq[i] > 0.5:
            activate = True
        if activate and aSeq[i] < 0.5:
            first_bellow_frame = i
            break
    return first_bellow_frame


def Smooth_sdt6(predict_sdt, threshold=0.5):
    # predict shape: (time step, 3)
    Filter = np.ndarray(shape=(5,), dtype=float, buffer=np.array([0.25, 0.5, 1.0, 0.5, 0.25]))
    # Filter = np.ndarray(shape=(5,), dtype=float, buffer=np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    sSeq = []
    dSeq = []
    onSeq = []
    offSeq = []

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
        return [],[],[],[],[],0

    Tpeaks = onpeaks + offpeaks
    Tpeaks.sort()

    intervalSD = [0]

    for i in range(len(Tpeaks) - 1):
        current_sd = 0 if sum(sSeq[Tpeaks[i]:Tpeaks[i + 1]]) > sum(dSeq[Tpeaks[i]:Tpeaks[i + 1]]) else 1
        intervalSD.append(current_sd)
    intervalSD.append(0)

    MissingT = 0
    AddingT = 0
    est_intervals = []
    t_idx = 0
    while t_idx < len(Tpeaks)-2:

        if t_idx == 0 and Tpeaks[t_idx] not in onpeaks:
            if intervalSD[0] == 1 and intervalSD[1] == 0:
                onset_inserted = find_first_bellow_thres(sSeq[0:Tpeaks[0]])
                if onset_inserted != Tpeaks[0] and Tpeaks > onset_inserted + 1:
                    est_intervals.append([0.02 * onset_inserted + 0.01, 0.02 * Tpeaks[0] + 0.01])
                    AddingT += 1
                else:
                    MissingT += 1
            t_idx += 1

        if Tpeaks[t_idx] in onpeaks and Tpeaks[t_idx + 1] in offpeaks:
            if Tpeaks[t_idx] == Tpeaks[t_idx + 1]:
                t_idx += 1
                continue
            if Tpeaks[t_idx + 1] > Tpeaks[t_idx] + 1:
                est_intervals.append([0.02 * Tpeaks[t_idx] + 0.01, 0.02 * Tpeaks[t_idx + 1] + 0.01])
            assert (Tpeaks[t_idx] < Tpeaks[t_idx + 1])
            t_idx += 2
            if t_idx > len(Tpeaks)-2:
                break
        elif Tpeaks[t_idx] in onpeaks and Tpeaks[t_idx + 1] in onpeaks:
            offset_inserted = find_first_bellow_thres(dSeq[Tpeaks[t_idx]:Tpeaks[t_idx + 1]]) + Tpeaks[t_idx]
            if offset_inserted != Tpeaks[t_idx] and offset_inserted > Tpeaks[t_idx] + 1:
                est_intervals.append([0.02 * Tpeaks[t_idx] + 0.01, 0.02 * offset_inserted + 0.01])
                AddingT += 1
                assert (Tpeaks[t_idx] < offset_inserted)
            else:
                MissingT += 1

            t_idx += 1
        elif Tpeaks[t_idx] in offpeaks:
            if intervalSD[t_idx] == 1 and intervalSD[t_idx + 1] == 0:
                onset_inserted = find_first_bellow_thres(sSeq[Tpeaks[t_idx - 1]:Tpeaks[t_idx]]) + Tpeaks[t_idx - 1]
                if onset_inserted != Tpeaks[t_idx - 1] and Tpeaks[t_idx] > onset_inserted + 1:
                    est_intervals.append([0.02 * onset_inserted + 0.01, 0.02 * Tpeaks[t_idx] + 0.01])
                    AddingT += 1
                    assert (onset_inserted < Tpeaks[t_idx])
                else:
                    MissingT += 1

            t_idx += 1

    # print("Missing ratio: ", MissingT/len(est_intervals))
    print("Conflict ratio: ", MissingT / (len(Tpeaks) + AddingT))

    # Modify 1
    sSeq_np = np.array(sSeq, dtype=float)
    dSeq_np = np.array(dSeq, dtype=float)
    onSeq_np = np.array(onSeq, dtype=float)
    offSeq_np = np.array(offSeq, dtype=float)


    # print(np.ndarray(shape=(len(est_intervals), 2), dtype=float,
    #                   buffer=np.array(est_intervals)),"\n", sSeq_np,"\n", dSeq_np,"\n", onSeq_np,"\n", offSeq_np,"\n", MissingT / (
    #                len(Tpeaks) + AddingT))
    # return np.ndarray(shape=(len(onset_times),), dtype=float, buffer=np.array(onset_times)), np.ndarray(shape=(len(offset_times),), dtype=float, buffer=np.array(offset_times)), sSeq_np, dSeq_np, tSeq_np
    return np.array(est_intervals, dtype=float), sSeq_np, dSeq_np, onSeq_np, offSeq_np, MissingT / (len(Tpeaks) + AddingT)


def get_Resnet():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    num_fout = model.conv1.out_channels
    model.conv1 = nn.Conv2d(9, num_fout, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                            bias=False)
    model.fc = nn.Linear(model.fc.in_features, 6)
    model.avgpool = nn.AvgPool2d(kernel_size=(6, 1), stride=1, padding=0)

    return model


def testset_evaluation(path, f_path, model=None, writer_in=None, timestep=None):
    if not model:
        model = get_Resnet()
    if not writer_in:
        writer = SummaryWriter()
    else:
        writer = writer_in

    if not timestep:
        timestep = 0

    wav_files = [os.path.join(path, file) for file in os.listdir(path) if '.wav' in file]
    labels = [os.path.join(path, label) for label in os.listdir(path) if '.notes.' in label]
    features = [os.path.join(f_path, features) for features in os.listdir(f_path) if '_FEAT' in features]

    sum_on_F1 = 0
    sum_off_F1 = 0

    model.eval()
    count = 0
    print("testing on testset for on/off_F1\n")
    for index in range(len(features)):
        if count > 4:  # shorten test time by sampling
            break
        record = []
        features_full = np.load(features[index])
        label_path = str(pathlib.Path(labels[index]).parent / (
                    pathlib.Path(features[index]).stem.split('.')[0] + ".notes.Corrected"))
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
        features_full = np.power(features_full/features_full.max(), hparam.gamma_mu) #normalize &gamma compression


        for test_step in range(features_full.shape[1] - 18):
            curr_clip = features_full[:, test_step:test_step + 19]
            curr_clip = torch.from_numpy(curr_clip)
            curr_clip = curr_clip.view(9, 174, -1).float()
            curr_clip = curr_clip.unsqueeze(0)
            curr_clip = curr_clip.to(device)
            model = model.to(device)
            out_label = model(curr_clip)
            out_label = out_label.squeeze(0).squeeze(0).cpu().detach().numpy()

            record.append(out_label)

        record = np.array(record)
        print(record.shape)
        est_intervals,_,_,_,_,_ = Smooth_sdt6(record)
        # est_labels = output2label(record, is_batch=False, is_nparray=True)
        # est_label_sec_on, est_label_sec_off = timestep2second(est_labels)
        est_smooth_label_sec_on=[]
        est_smooth_label_sec_off=[]
        for interval in est_intervals:
            est_smooth_label_sec_on.append(interval[0])
            est_smooth_label_sec_off.append(interval[1])
        est_smooth_label_sec_on = np.array(est_smooth_label_sec_on)
        est_smooth_label_sec_off = np.array(est_smooth_label_sec_off)

        on_F, on_P, on_R = mir_eval.onset.f_measure(gt_label_sec_on, est_smooth_label_sec_on)
        off_F, off_P, off_R = mir_eval.onset.f_measure(gt_label_sec_off, est_smooth_label_sec_off)

        sum_on_F1 += on_F
        sum_off_F1 += off_F
        count += 1
        print(f"smooth_on_F1: {on_F}, smooth_off_F1: {off_F}")


    writer.add_scalars(f"scalar\\onoff_F1", {'on_F1': sum_on_F1 / count, 'off_F1': sum_off_F1 / count}, timestep)


def whole_song_sampletest(path, f_path, model=None, writer_in=None, timestep=None):
    if not model:
        model = get_Resnet()
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
    for index in range(len(features)):
        record = []

        features_full = np.load(features[index])

        a = features[index]
        label_path = str(pathlib.Path(labels[index]).parent / (
                    pathlib.Path(features[index]).stem.split('.')[0] + ".notes.Corrected"))
        if index ==1:
            features_full = np.load("data/train/Process_data/FEAT/01-D_AMairena.wav_FEAT.npy") # if trainset isok?
            label_path = "data/train/TONAS/Deblas/01-D_AMairena.notes.Corrected"

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
        features_full = np.power(features_full/features_full.max(), hparam.gamma_mu) #normalize &gamma compression

        for test_step in range(features_full.shape[1] - 18):
            curr_clip = features_full[:, test_step:test_step + 19]
            curr_clip = torch.from_numpy(curr_clip)
            curr_clip = curr_clip.view(9, 174, -1).float()
            curr_clip = curr_clip.unsqueeze(0)
            curr_clip = curr_clip.to(device)
            model = model.to(device)
            out_label = model(curr_clip)
            out_label = out_label.squeeze(0).squeeze(0).cpu().detach().numpy()

            record.append(out_label)
            # if len(record) > hparam.whole_song_max_len:
            #     print(f"{label_path} OK")
            #     break
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

    # for batch_idx in range(ref_label.shape[0]):
    #     for frame_idx in range(ref_label.shape[1]):
    #         norm_sa = est_label[batch_idx][frame_idx][0]+est_label[batch_idx][frame_idx][1]
    #         norm_on = est_label[batch_idx][frame_idx][2]+est_label[batch_idx][frame_idx][3] # make sure the sum of on and Xon =1
    #         norm_off = est_label[batch_idx][frame_idx][4]+est_label[batch_idx][frame_idx][5]
    #         est_label[batch_idx][frame_idx][0]/=norm_sa
    #         est_label[batch_idx][frame_idx][1]/=norm_sa
    #         est_label[batch_idx][frame_idx][2]/=norm_on
    #         est_label[batch_idx][frame_idx][3]/=norm_on
    #         est_label[batch_idx][frame_idx][4]/=norm_off
    #         est_label[batch_idx][frame_idx][5]/=norm_off
    #
    #
    # est_label = (est_label > hparam.label_threshold).int()
    est_label = output2label(est_label)
    ref_label = ref_label.int()

    for batch_idx in range(ref_label.shape[0]):
        for frame_idx in range(ref_label.shape[1]):
            if torch.equal(est_label[batch_idx][frame_idx], ref_label[batch_idx][frame_idx]):
                correct += 1

    return correct / total


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

        # tail = note[0] // 0.02 * 0.02 + 0.02
        tail = len(timestep) * 0.02
        end_tail = (note[0] + note[1]) // 0.02 * 0.02 + 0.02
        status = [0, 1, 0, 1, 0, 1]
        ccc = ((note[0] + note[1] - tail) / 0.02)
        for _ in range(int((note[0] + note[1] - tail + 1e-4) // 0.02)):
            timestep.append(status)
            pitch.append(note[2])

        status = [0, 1, 0, 1, 1, 0]
        timestep.append(status)
        pitch.append(note[2])
        # print(len(timestep), len(pitch))

    return timestep, pitch


if __name__ == '__main__':
    path = "data/test/EvaluationFramework_ISMIR2014/DATASET"
    f_path = "data/test/Process_data/FEAT"

    model= get_Resnet().to(device)
    model.load_state_dict(torch.load("checkpoint/1920.pth"))
    print("load OK")

    # testset_evaluation(path, f_path, model=model)

    testsample_path = hparam.testsample_path
    testsample_f_path = hparam.testsample_f_path
    whole_song_sampletest(testsample_path, testsample_f_path, model=model)

    # for file in os.listdir('data/train/TONAS/Deblas/'):
    #     if '.notes.Corrected' in file:
    #         dir = f'data/train/TONAS/Deblas/{file}'
    #         notes = read_notefile(dir)
    #         aa,pp = note2timestep(notes)
    #
    #         print(((notes[-1][0]+notes[-1][1]+1e-4)//0.02+1)*0.02, len(aa)*0.02,file)
    #         assert ((notes[-1][0]+notes[-1][1]+1e-4)//0.02+1)*0.02==len(aa)*0.02
    #
    #         aa = np.array(aa)
    #         pp = np.array(pp)

    # dir = f'data/train/TONAS/Deblas/52-M1_ManueldeAngustias.notes.Corrected'
    # notes = read_notefile(dir)
    # aa,pp = note2timestep(notes)
    # aa = np.array(aa)
    # pp = np.array(pp)
