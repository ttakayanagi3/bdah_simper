import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F


def label_distance(labels_1, labels_2, dist_fn='l1', label_temperature=0.1):
    # soft_max = nn.Softmax(dim=1)
    if dist_fn == 'l1':
        dist_mat = -torch.abs(labels_1[:, :, None] - labels_2[:, None, :])
    else:
        dist_mat = -torch.abs(labels_1[:, :, None] - labels_2[:, None, :]) ** 2

    prob_mat = F.softmax(dist_mat / label_temperature, dim=-1)

    return prob_mat


def max_cross_corr(feats_1, feats_2):
    feats_2 = feats_2.to(feats_1.dtype)
    feats_1 = feats_1 - torch.mean(feats_1, dim=-1, keepdim=True)
    feats_2 = feats_2 - torch.mean(feats_2, dim=-1, keepdim=True)
    # feats_1 = feats_1 - torch.mean(feats_1)
    # feats_2 = feats_2 - torch.mean(feats_2)

    min_N = min(feats_1.shape[-1], feats_2.shape[-1])
    padded_N = max(feats_1.shape[-1], feats_2.shape[-1]) * 2
    pad_1 = torch.zeros(feats_1.shape[0], padded_N - feats_1.shape[-1])
    pad_2 = torch.zeros(feats_2.shape[0], padded_N - feats_2.shape[-1])

    feats_1_pad = torch.cat((feats_1, pad_1), dim=-1)
    feats_2_pad = torch.cat((feats_2, pad_2), dim=-1)

    feats_1_fft = fft.rfft(feats_1_pad)
    feats_2_fft = fft.rfft(feats_2_pad)
    X = feats_1_fft * torch.conj(feats_2_fft)

    power_norm = (torch.std(feats_1, dim=-1, keepdim=True) * torch.std(feats_2, dim=-1, keepdim=True)).to(X.dtype)
    # power_norm = (torch.std(feats_1, keepdim=True) * torch.std(feats_2, keepdim=True))
    power_norm = torch.where(power_norm == 0, torch.ones_like(power_norm), power_norm)
    X = X / power_norm

    cc = fft.irfft(X) / (min_N - 1)
    max_cc = torch.max(cc, dim=-1).values

    return max_cc


if __name__ == '__main__':
    batch_size = 4
    label1 = torch.randn((batch_size, 10))
    label2 = torch.randn((batch_size, 10))
    labels = label_distance(label1, label2)
    print(labels)