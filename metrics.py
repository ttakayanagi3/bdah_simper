import torch
import torch.nn as nn


def label_distance(labels_1, labels_2, dist_fn='l1', label_temperature=0.1):
    soft_max = nn.Softmax(dim=1)
    if dist_fn == 'l1':
        dist_mat = -torch.abs(labels_1 - labels_2)
    else:
        dist_mat = -torch.abs(labels_1 - labels_2) ** 2

    prob_mat = soft_max(dist_mat / label_temperature)

    return prob_mat


if __name__ =='__main__':
    batch_size = 4
    label1 = torch.randn((batch_size, 10))
    label2 = torch.randn((batch_size, 10))
    labels = label_distance(label1, label2)
    print(labels)