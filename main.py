import os
import argparse

import torch
import torch.nn as nn
from torchvision import datasets as ds
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import preprocess as p
import models as m
import metrics as me
import trainer as t

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def load_mnist(opt, train_dype='train', base_dir='./data', need_sampling=True,
               data_size=1000):
    os.makedirs(base_dir, exist_ok=True)
    mnist = ds.MNIST(root=base_dir, train=True,
                     download=True)
    #
    # random sampling
    #
    if need_sampling:

        N = mnist.train_data.shape[0]
        indicies = torch.randint(0, N, (data_size,))
        if train_dype == 'train':
            sampled_data = mnist.train_data[indicies]
            sampled_label = mnist.train_labels[indicies]
        else:
            sampled_data = mnist.test_data[indicies]
            sampled_label = mnist.test_labels[indicies]
    else:
    #
    # in this case, all data are used
    #
        if train_dype == 'train':
            sampled_data = mnist.train_data[:]
            sampled_label = mnist.train_labels[:]
        else:
            sampled_data = mnist.test_data[:]
            sampled_label = mnist.test_labels[:]
    #
    # generate frequency
    #
    freq = torch.rand(data_size) * (opt.max_freq - opt.min_freq) + opt.min_freq

    return (sampled_data, sampled_label, freq)


def main():
    FPS = 30
    LENGTH_SEC = 5
    NUM_FRAMES = FPS * LENGTH_SEC
    MAX_SPEED = 3
    IMG_SIZE = 28
    CHANNELS = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--min_freq", type=float, default=0.5, help="minimum frequency")
    parser.add_argument("--max_freq", type=float, default=5, help="minimum frequency")
    parser.add_argument("--random_phase", type=int, default=1, help="1: apply random phase, 0: default")
    parser.add_argument("--FPS", type=int, default=FPS)
    parser.add_argument("--LENGTH_SEC", type=int, default=LENGTH_SEC)
    parser.add_argument("--batch_size", type=int, default=4, help="size of mini-batches")
    parser.add_argument("--n_epochs", type=int, default=10, help="epochs")
    parser.add_argument("--NUM_SELF_CON_SIMPER", type=int, default=10)
    parser.add_argument("--MAX_SPEED", type=int, default=MAX_SPEED)
    parser.add_argument("--SSL_FRAMES", type=int, default=NUM_FRAMES // MAX_SPEED)
    parser.add_argument("--IMG_SIZE", type=int, default=IMG_SIZE)
    parser.add_argument("--CHANNELS", type=int, default=CHANNELS)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--DEBUG", type=int, default=0)
    #
    #
    parser.add_argument("--label_dist_fn", type=str, default='l1')
    parser.add_argument("--feat_dist_fn", type=str, default='max_corr')
    parser.add_argument("--label_temperature", type=float, default=0.1)

    opt = parser.parse_args()
    print(opt)

    train_data, train_label, train_freq = load_mnist(opt, train_dype='train')
    dataset = p.CustomDataset(train_data, train_label, train_freq, opt)
    # dataset = p.CustomDataset(train_data, train_label, train_freq, opt, debug=True)
    data_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size)
    model = m.SimPer(opt)
    if opt.DEBUG == 1:
        import seaborn as sns
        import matplotlib.pyplot as plt
        need_visualize = False

        data_iter = iter(data_loader)
        frames, all_speed, y_angle = next(data_iter)
        if need_visualize:
            frames_numpy = frames.numpy()[0, 0]
            for i in range(5):
                sns.heatmap(frames_numpy[i])
                plt.show()

        #
        # split all speed
        # 2 * M -> M, M
        #
        num_arguments = frames.shape[1]
        half_of_num_arguments = int(num_arguments // 2)
        all_speed1 = all_speed[:, :half_of_num_arguments]
        all_speed2 = all_speed[:, half_of_num_arguments:]
        #
        # label distance
        #
        all_labels = me.label_distance(all_speed1, all_speed2,
                                       opt.label_dist_fn, opt.label_temperature)
        mini_batch_size = frames.shape[0]
        shape = frames.shape[2:]
        transform_shape = (mini_batch_size * num_arguments, *shape)
        frames_transformed = frames.view(transform_shape)
        all_z = model(frames_transformed, 'f')
        all_z = all_z.view(mini_batch_size, num_arguments, -1)

        for feats, labels in zip(all_z, all_labels):
            print(feats, labels)
            feat1 = feats[:half_of_num_arguments]
            feat2 = feats[half_of_num_arguments:]
            if opt.feat_dist_fn == 'max_corr':
                feat_dist = me.max_cross_corr(feat1, feat2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    t.train(data_loader, model, criterion, optimizer, opt)



    print('complete')
    return
#


if __name__ == '__main__':
    main()
