import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def label_distance(labels_1, labels_2, debug, dist_fn='l1', label_temperature=0.1):
    # soft_max = nn.Softmax(dim=1)
    if dist_fn == 'l1':
        dist_mat = -torch.abs(labels_1[:, :, None] - labels_2[:, None, :])
    else:
        dist_mat = -torch.abs(labels_1[:, :, None] - labels_2[:, None, :]) ** 2

    prob_mat = F.softmax(dist_mat / label_temperature, dim=-1)
    if debug == 1:
        prob_mat_numpy = prob_mat.detach().cpu().numpy()
    true_index = torch.argmax(prob_mat, dim=1)
    # true_index_numpy = true_index.detach().cpu().numpy()

    return prob_mat, true_index


# def max_cross_corr(feats_1, feats_2, device):
#     feats_2 = feats_2.to(feats_1.dtype)
#     feats_1 = feats_1 - torch.mean(feats_1, dim=-1, keepdim=True)
#     feats_2 = feats_2 - torch.mean(feats_2, dim=-1, keepdim=True)
#     # feats_1 = feats_1 - torch.mean(feats_1)
#     # feats_2 = feats_2 - torch.mean(feats_2)
#
#     min_N = min(feats_1.shape[-1], feats_2.shape[-1])
#     padded_N = max(feats_1.shape[-1], feats_2.shape[-1]) * 2
#     pad_1 = torch.zeros(feats_1.shape[0], padded_N - feats_1.shape[-1])
#     pad_2 = torch.zeros(feats_2.shape[0], padded_N - feats_2.shape[-1])
#
#     pad_1 = pad_1.to(device)
#     pad_2 = pad_2.to(device)
#
#     feats_1_pad = torch.cat((feats_1, pad_1), dim=-1)
#     feats_2_pad = torch.cat((feats_2, pad_2), dim=-1)
#
#     feats_1_fft = fft.rfft(feats_1_pad)
#     feats_2_fft = fft.rfft(feats_2_pad)
#     X = feats_1_fft * torch.conj(feats_2_fft)
#
#     power_norm = (torch.std(feats_1, dim=-1, keepdim=True) * torch.std(feats_2, dim=-1, keepdim=True)).to(X.dtype)
#     # power_norm = (torch.std(feats_1, keepdim=True) * torch.std(feats_2, keepdim=True))
#     power_norm = torch.where(power_norm == 0, torch.ones_like(power_norm), power_norm)
#     X = X / power_norm
#
#     cc = fft.irfft(X) / (min_N - 1)
#     max_cc = torch.max(cc, dim=-1).values
#
#     return max_cc

def _max_cross_corr(feats_1, feats_2, device):
    # feats_1: 1 x T(# time stamp)
    # feats_2: M(# aug) x T(# time stamp)
    feats_2 = feats_2.to(feats_1.dtype)
    feats_1 = feats_1 - torch.mean(feats_1, dim=-1, keepdim=True)
    feats_2 = feats_2 - torch.mean(feats_2, dim=-1, keepdim=True)

    min_N = min(feats_1.shape[-1], feats_2.shape[-1])
    padded_N = max(feats_1.shape[-1], feats_2.shape[-1]) * 2
    feats_1_pad = F.pad(feats_1, (0, padded_N - feats_1.shape[-1]))
    feats_2_pad = F.pad(feats_2, (0, padded_N - feats_2.shape[-1]))

    feats_1_fft = torch.fft.rfft(feats_1_pad)
    feats_2_fft = torch.fft.rfft(feats_2_pad)
    X = feats_1_fft * torch.conj(feats_2_fft)

    power_norm = (torch.std(feats_1, dim=-1, keepdim=True) *
                  torch.std(feats_2, dim=-1, keepdim=True)).to(X.dtype)
    power_norm = torch.where(power_norm == 0, torch.ones_like(power_norm), power_norm)
    X = X / power_norm

    cc = torch.fft.irfft(X) / (min_N - 1)
    max_cc = torch.max(cc, dim=-1).values

    return max_cc


def batched_max_cross_corr(x, y, device):
    # x: M(# aug) x T(# time stamp)
    # y: M(# aug) x T(# time stamp)
    per_x_dist = lambda i: _max_cross_corr(x[i:(i+1), :], y, device)
    dist = torch.stack([per_x_dist(i) for i in range(x.shape[0])])
    return dist


if __name__ == '__main__':
    batch_size = 4
    label1 = torch.randn((batch_size, 10))
    label2 = torch.randn((batch_size, 10))
    debug = 1
    labels, label_index = label_distance(label1, label2, debug)
    prob_mat_numpy = labels.detach().cpu().numpy()

    print(labels)

    device = 'cpu'

    fs = 1000
    t = np.arange(0, 1, 1 / fs)
    target_length = 200

    #
    # original sin waves
    #
    freq1 = 5
    sin_wave1 = np.sin(2 * np.pi * freq1 * t)
    sin_wave1 = sin_wave1[:target_length]
    freq2 = 20
    sin_wave2 = np.sin(2 * np.pi * freq2 * t)
    sin_wave2 = sin_wave2[:target_length]
    freq3 = 50
    sin_wave3 = np.sin(2 * np.pi * freq3 * t)
    sin_wave3 = sin_wave3[:target_length]

    sin_arr = np.stack([sin_wave1, sin_wave2, sin_wave3])

    #
    # shifted sin waves
    #
    shift = 5
    freq1 = 5
    sin_wave1 = np.sin(2 * np.pi * freq1 * t + shift)
    sin_wave1 = sin_wave1[:target_length]
    freq2 = 20
    sin_wave2 = np.sin(2 * np.pi * freq2 * t + shift)
    sin_wave2 = sin_wave2[:target_length]
    freq3 = 50
    sin_wave3 = np.sin(2 * np.pi * freq3 * t + shift)
    sin_wave3 = sin_wave3[:target_length]

    sin_arr2 = np.stack([sin_wave1, sin_wave2, sin_wave3])

    plt.plot(sin_arr[0, :])
    plt.plot(sin_arr2[0, :])
    plt.show()

    sin_tensor1 = torch.FloatTensor(sin_arr)
    sin_tensor2 = sin_tensor1.clone()
    feat_dist = batched_max_cross_corr(sin_tensor1, sin_tensor2, device)
    print(feat_dist)

    sin_tensor3 = torch.roll(sin_tensor2, shifts=2, dims=1)
    feat_dist = batched_max_cross_corr(sin_tensor1, sin_tensor3, device)
    print(feat_dist)

    print('Complete')