# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 21:54:18 2017

@author: lisu
"""
import soundfile as sf
import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
import librosa
import argparse
import hparam
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numba as nb
from utils import read_notefile, note2timestep



def STFT(x, fr, fs, Hop, h):
    t = np.arange(Hop, np.ceil(len(x) / float(Hop)) * Hop, Hop)
    N = int(fs / float(fr))
    window_size = len(h)
    f = fs * np.linspace(0, 0.5, int(np.round(N / 2)), endpoint=True)
    Lh = int(np.floor(float(window_size - 1) / 2))
    tfr = np.zeros((int(N), len(t)), dtype=np.float)

    for icol in range(0, len(t)):
        ti = int(t[icol])
        tau = np.arange(int(-min([round(N / 2.0) - 1, Lh, ti - 1])), \
                        int(min([round(N / 2.0) - 1, Lh, len(x) - ti])))
        indices = np.mod(N + tau, N) + 1
        tfr[indices - 1, icol] = x[ti + tau - 1] * h[Lh + tau - 1] \
                                 / np.linalg.norm(h[Lh + tau - 1])

    tfr = abs(scipy.fftpack.fft(tfr, n=N, axis=0))
    return tfr, f, t, N


def nonlinear_func(X, g, cutoff):
    cutoff = int(cutoff)
    if g != 0:
        X[X < 0] = 0
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
        X = np.power(X, g)
    else:
        X = np.log(X)
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
    return X


def Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1 / tc
    Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break

    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest - 1, len(f)), dtype=np.float)
    for i in range(1, Nest - 1):
        l = int(round(central_freq[i - 1] / fr))
        r = int(round(central_freq[i + 1] / fr) + 1)
        # rounding1
        if l >= r - 1:
            freq_band_transformation[i, l] = 1
        else:
            for j in range(l, r):
                if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                    freq_band_transformation[i, j] = (f[j] - central_freq[i - 1]) / (
                                central_freq[i] - central_freq[i - 1])
                elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                    freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (
                                central_freq[i + 1] - central_freq[i])
    tfrL = np.dot(freq_band_transformation, tfr)
    return tfrL, central_freq


def Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1 / tc
    Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    f = 1 / q
    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest - 1, len(f)), dtype=np.float)
    for i in range(1, Nest - 1):
        for j in range(int(round(fs / central_freq[i + 1])), int(round(fs / central_freq[i - 1]) + 1)):
            if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                freq_band_transformation[i, j] = (f[j] - central_freq[i - 1]) / (central_freq[i] - central_freq[i - 1])
            elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])

    tfrL = np.dot(freq_band_transformation, ceps)
    return tfrL, central_freq


def CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave):
    NumofLayer = np.size(g)

    [tfr, f, t, N] = STFT(x, fr, fs, Hop, h)
    tfr = np.power(abs(tfr), g[0])
    tfr0 = tfr  # original STFT
    ceps = np.zeros(tfr.shape)


    if NumofLayer >= 2:
        for gc in range(1, NumofLayer):
            if np.remainder(gc, 2) == 1:
                tc_idx = round(fs * tc)
                ceps = np.real(np.fft.fft(tfr, axis=0)) / np.sqrt(N)
                ceps = nonlinear_func(ceps, g[gc], tc_idx)
            else:
                fc_idx = round(fc / fr)
                tfr = np.real(np.fft.fft(ceps, axis=0)) / np.sqrt(N)
                tfr = nonlinear_func(tfr, g[gc], fc_idx)

    tfr0 = tfr0[:int(round(N / 2)), :]
    tfr = tfr[:int(round(N / 2)), :]
    ceps = ceps[:int(round(N / 2)), :]

    HighFreqIdx = int(round((1 / tc) / fr) + 1)
    f = f[:HighFreqIdx]
    tfr0 = tfr0[:HighFreqIdx, :]
    tfr = tfr[:HighFreqIdx, :]
    HighQuefIdx = int(round(fs / fc) + 1)
    q = np.arange(HighQuefIdx) / float(fs)
    ceps = ceps[:HighQuefIdx, :]

    tfrL0, central_frequencies = Freq2LogFreqMapping(tfr0, f, fr, fc, tc, NumPerOctave)
    tfrLF, central_frequencies = Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOctave)
    tfrLQ, central_frequencies = Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOctave)



    return tfrL0, tfrLF, tfrLQ, f, q, t, central_frequencies



def full_feature_extraction(x, window_size, label_note=None):


    # if x.shape[1] > 1:
    #     x = np.mean(x, axis=1)
    # x = scipy.signal.resample_poly(x, 16000, fs)
    fs = 16000.0  # sampling frequency
    x = x.astype('float32')
    Hop = 320  # hop size (in sample)

    h = [scipy.signal.blackmanharris(w_size) for w_size in window_size]

    h3 = scipy.signal.blackmanharris(743)  # window size - 2048   (186, 372, 743)
    h2 = scipy.signal.blackmanharris(372)  # window size - 1024
    h1 = scipy.signal.blackmanharris(186)  # window size - 512
    fr = 2.0  # frequency resolution
    fc = 80.0  # the frequency of the lowest pitch
    tc = 1 / 1000.0  # the period of the highest pitch
    g = np.array([0.24, 0.6, 1])
    NumPerOctave = 48  # Number of bins per octave


    Features_CFP = [ CFP_filterbank(x, fr, fs, Hop, sub_h, fc, tc, g, NumPerOctave) for sub_h in h  ]

    Z_list=[]
    ZN_list=[]
    SN_list=[]
    SIN_list=[]

    for Feature in Features_CFP:
        Z_list.append(Feature[1]*Feature[2])
        ZN_list.append((Z_list[-1] - np.mean(Z_list[-1])) / np.std(Z_list[-1]))
        SN_list.append(gen_spectral_flux(Feature[0], invert=False, norm=True))
        SIN_list.append(gen_spectral_flux(Feature[0], invert=True, norm=True))

    All_Z =Z_list[0]
    All_ZN=ZN_list[0]
    All_SN= SN_list[0]
    All_SIN = SIN_list[0]

    for idx in range(len(Z_list)-1):
        All_Z = np.concatenate((All_Z, Z_list[idx+1]), axis=0)
        All_ZN = np.concatenate((All_ZN, ZN_list[idx+1]), axis=0)
        All_SN = np.concatenate((All_SN, SN_list[idx+1]), axis=0)
        All_SIN = np.concatenate((All_SIN, SIN_list[idx+1]), axis=0)


    new_SN_SIN_ZN = np.concatenate((All_SN, All_SIN, All_ZN), axis=0)


    return new_SN_SIN_ZN



def gen_spectral_flux(S, invert=False, norm=True):
    flux = np.diff(S)
    first_col = np.zeros((S.shape[0], 1))
    flux = np.hstack((first_col, flux))

    if invert:
        flux = flux * (-1.0)

    flux = np.where(flux < 0, 0.0, flux)

    if norm:
        flux = (flux - np.mean(flux)) / np.std(flux)

    return flux


def feature_extraction(filename):
    x, fs = librosa.load(filename, sr = hparam.sr)
    fs = 16000.0  # sampling frequency
    x = x.astype('float32')
    Hop = 320  # hop size (in sample)
    h = scipy.signal.blackmanharris(2049)  # window size
    fr = 2.0  # frequency resolution
    fc = 80.0  # the frequency of the lowest pitch
    tc = 1 / 1000.0  # the period of the highest pitch
    g = np.array([0.24, 0.6, 1])
    NumPerOctave = 48  # Number of bins per octave

    tfrL0, tfrLF, tfrLQ, f, q, t, CenFreq = CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave)
    Z = tfrLF * tfrLQ
    return Z, t, CenFreq, tfrL0, tfrLF, tfrLQ


def patch_extraction(Z, patch_size, th):
    # Z is the input spectrogram or any kind of time-frequency representation
    M, N = np.shape(Z)
    half_ps = int(np.floor(float(patch_size) / 2))

    Z = np.append(np.zeros([M, half_ps]), Z, axis=1)
    Z = np.append(Z, np.zeros([M, half_ps]), axis=1)
    Z = np.append(Z, np.zeros([half_ps, N + 2 * half_ps]), axis=0)

    M, N = np.shape(Z)

    data = np.zeros([300000, patch_size, patch_size])
    mapping = np.zeros([300000, 2])
    counter = 0
    for t_idx in range(half_ps, N - half_ps):
        PKS, LOCS = findpeaks(Z[:, t_idx], th)
        for mm in range(0, len(LOCS)):
            if LOCS[mm] >= half_ps and LOCS[mm] < M - half_ps and counter < 300000:  # and PKS[mm]> 0.5*max(Z[:,t_idx]):
                patch = Z[np.ix_(range(LOCS[mm] - half_ps, LOCS[mm] + half_ps + 1),
                                 range(t_idx - half_ps, t_idx + half_ps + 1))]
                patch = patch.reshape(1, patch_size, patch_size)

                data[counter, :, :] = patch
                mapping[counter, :] = np.array([[LOCS[mm], t_idx]])
                counter = counter + 1
            elif LOCS[mm] >= half_ps and LOCS[mm] < M - half_ps and counter >= 300000:
                print('Out of the biggest size. Please shorten the input audio.')

    data = data[:counter - 1, :, :]
    mapping = mapping[:counter - 1, :]
    Z = Z[:M - half_ps, :]

    return data, mapping, half_ps, N, Z



def contour_prediction(mapping, pred, N, half_ps, Z, t, CenFreq, max_method):
    PredContour = np.zeros(N)

    pred = pred[:, 1]
    pred_idx = np.where(pred > 0.5)
    MM = mapping[pred_idx[0], :]

    pred_prob = pred[pred_idx[0]]

    MM = np.append(MM, np.reshape(pred_prob, [len(pred_prob), 1]), axis=1)
    MM = MM[MM[:, 1].argsort()]

    for t_idx in range(half_ps, N - half_ps):
        Candidate = MM[np.where(MM[:, 1] == t_idx)[0], :]

        if Candidate.shape[0] >= 2:
            if max_method == 'posterior':
                fi = np.where(Candidate[:, 2] == np.max(Candidate[:, 2]))
                fi = fi[0]
            elif max_method == 'prior':
                fi = Z[Candidate[:, 0].astype('int'), t_idx].argmax(axis=0)
            fi = fi.astype('int')

            PredContour[Candidate[fi, 1].astype('int')] = Candidate[fi, 0]
        elif Candidate.shape[0] == 1:
            PredContour[Candidate[0, 1].astype('int')] = Candidate[0, 0]

            # clip the padding of time
    PredContour = PredContour[range(half_ps, N - half_ps)]

    for k in range(len(PredContour)):
        if PredContour[k] > 1:
            PredContour[k] = CenFreq[PredContour[k].astype('int')]


    result = np.zeros([t.shape[0], 2])
    result[:, 0] = t / 16000.0
    result[:, 1] = PredContour
    return result


def contour_pred_from_raw(Z, t, CenFreq):
    PredContour = Z.argmax(axis=0)
    for k in range(len(PredContour)):
        if PredContour[k] > 1:
            PredContour[k] = CenFreq[PredContour[k].astype('int')]
    result = np.zeros([t.shape[0], 2])
    result[:, 0] = t / 16000.0
    result[:, 1] = PredContour
    return result


def show_prediction(mapping, pred, N, half_ps, Z, t):
    postgram = np.zeros(Z.shape)
    pred = pred[:, 1]
    for i in range(pred.shape[0]):
        postgram[mapping[i, 0].astype('int'), mapping[i, 1].astype('int')] = pred[i]
    return postgram


def findpeaks(x, th):
    # x is an input column vector
    M = x.shape[0]
    pre = x[1:M - 1] - x[0:M - 2]
    pre[pre < 0] = 0
    pre[pre > 0] = 1

    post = x[1:M - 1] - x[2:]
    post[post < 0] = 0
    post[post > 0] = 1

    mask = pre * post
    ext_mask = np.append([0], mask, axis=0)
    ext_mask = np.append(ext_mask, [0], axis=0)

    pdata = x * ext_mask
    pdata = pdata - np.tile(th * np.amax(pdata, axis=0), (M, 1))
    pks = np.where(pdata > 0)
    pks = pks[0]

    locs = np.where(ext_mask == 1)
    locs = locs[0]
    return pks, locs


def melody_extraction(infile, outfile, model=None):

    print('Feature Extraction: Extracting Pitch Contour ...')
    Z, t, CenFreq, tfrL0, tfrLF, tfrLQ = feature_extraction(infile)
    return  Z, tfrL0, tfrLF, tfrLQ, t, CenFreq


# def output_feature_extraction(infile, outfile_z, outfile_t, outfile_f, outfile_s):

def output_feature_extraction(x, outfile_feat, outfile_z, outfile_cf, label_note=None, window_size=[743]):
    print('Feature Extraction: Extracting Spectral Difference and CFP ...')
    # Z, t, f, CenFreq, tfrL0, tfrLF, tfrLQ = full_feature_extraction(infile)
    SN_SIN_ZN= full_feature_extraction(x, label_note= label_note,window_size=window_size )

    np.save(outfile_feat, SN_SIN_ZN)



    return SN_SIN_ZN

def output_feature_extraction_nosave(x, window_size):
    print('Feature Extraction: Extracting Spectral Difference and CFP ...')
    # Z, t, f, CenFreq, tfrL0, tfrLF, tfrLQ = full_feature_extraction(infile)
    SN_SIN_ZN = full_feature_extraction(x, window_size)

    return SN_SIN_ZN

def librosa_HPSS(stft, mask=False):
    H, P = librosa.decompose.hpss(stft, kernel_size=(9,9))

    return H, P


if __name__ == "__main__":

    test_wav_dir = "data/test/EvaluationFramework_ISMIR2014/DATASET"

    test_tar_dir = "data/test/Process_data_S3_TEST"


    testsample_wav_dir = "data/test_sample/wav_label"
    testsample_tar_dir = "data/test_sample/Process_data_S3_TEST"

    train_wav_dir = "data/train/train_extension"
    train_tar_dir = "data/train/train_extension_S3_TEST"

    os.makedirs(test_tar_dir, exist_ok=True)
    os.makedirs(testsample_tar_dir, exist_ok=True)
    os.makedirs(train_tar_dir, exist_ok=True)



    for wav_dir, tar_dir in [(test_wav_dir, test_tar_dir),(testsample_wav_dir, testsample_tar_dir),(train_wav_dir, train_tar_dir)]:
        print(wav_dir)

        for wavfile in tqdm(os.listdir(wav_dir)) :

            if ".wav" in wavfile and not os.path.isfile(os.path.join(tar_dir,"FEAT" ,  f"{wavfile[:]}_FEAT.npy")):
                InFile = os.path.join(wav_dir, wavfile)
                os.makedirs(os.path.join(tar_dir,"FEAT"), exist_ok=True)
                OutFile_FEAT = os.path.join(tar_dir,"FEAT" ,  f"{wavfile[:]}_FEAT.npy")
                os.makedirs(os.path.join(tar_dir, "Z"), exist_ok=True)
                OutFile_Z = os.path.join(tar_dir, "Z", f"{wavfile[:]}_Z.npy")
                os.makedirs(os.path.join(tar_dir, "CF"), exist_ok=True)
                OutFile_CF = os.path.join(tar_dir, "CF", f"{wavfile[:]}_CF.npy")
                os.makedirs(os.path.join(tar_dir, "P"), exist_ok=True)
                OutFile_P = os.path.join(tar_dir, "P", f"{wavfile[:]}_P.npy")

                # melody_extraction(InFile, OutFile_P)

                x, fs = librosa.load(InFile, sr = hparam.sr)

                label_path = InFile[:-4]+".notes.Corrected"
                label_note = read_notefile(label_path)
                label_note, label_pitch = note2timestep(label_note)

                label_note = np.array(label_note)[:,2]


                output_feature_extraction(x, OutFile_FEAT, OutFile_Z, OutFile_CF, label_note=label_note, window_size=[743, 372, 186])
                print(InFile)




        print("finish!")
