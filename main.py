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
import loss as l


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def load_mnist(opt, train_dype='train', base_dir='./data', need_sampling=True,
               data_size=1000, target_digit=None):
    os.makedirs(base_dir, exist_ok=True)
    mnist = ds.MNIST(root=base_dir, train=True,
                     download=True)
    if target_digit is not None:
        train_data = mnist.train_data
        train_labels = mnist.train_labels
        train_data_numpy = train_data.cpu().numpy()

        sampled_data = []
        sampled_label = []
        for data, label in zip(train_data, train_labels):
            if label == target_digit:
                sampled_data.append(data)
                sampled_label.append(label)
            if len(sampled_data) == data_size:
                break
        sampled_data = torch.stack(sampled_data)
        sampled_label = torch.stack(sampled_label)
    else:
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
    # FPS = 30
    # FPS = 50
    FPS = 60
    LENGTH_SEC = 5
    NUM_FRAMES = FPS * LENGTH_SEC
    MAX_SPEED = 3
    IMG_SIZE = 28
    CHANNELS = 1
    UMAP = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--min_freq", type=float, default=0.5, help="minimum frequency")
    parser.add_argument("--max_freq", type=float, default=5, help="minimum frequency")
    parser.add_argument("--random_phase", type=int, default=1, help="1: apply random phase, 0: default")
    parser.add_argument("--FPS", type=int, default=FPS)
    parser.add_argument("--LENGTH_SEC", type=int, default=LENGTH_SEC)
    # parser.add_argument("--batch_size", type=int, default=4, help="size of mini-batches")
    if UMAP:
        parser.add_argument("--batch_size", type=int, default=2, help="size of mini-batches")
    else:
        parser.add_argument("--batch_size", type=int, default=4, help="size of mini-batches")
    # parser.add_argument("--batch_size", type=int, default=8, help="size of mini-batches")
    parser.add_argument("--n_epochs", type=int, default=15, help="epochs")
    parser.add_argument("--NUM_SELF_CON_SIMPER", type=int, default=10)
    parser.add_argument("--MAX_SPEED", type=int, default=MAX_SPEED)
    parser.add_argument("--SSL_FRAMES", type=int, default=NUM_FRAMES // MAX_SPEED)
    parser.add_argument("--IMG_SIZE", type=int, default=IMG_SIZE)
    parser.add_argument("--CHANNELS", type=int, default=CHANNELS)
    parser.add_argument("--extract_time_frames", type=int, default=90)
    # parser.add_argument("--lr", type=float, default=2e-3, help="Learning Rate")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning Rate")
    parser.add_argument("--DEBUG", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    # parser.add_argument("--experiment_name", type=str, default='FPS 50, Seq Length 80, Lr 1e-3, Batch size 4')
    # parser.add_argument("--experiment_name", type=str, default='FPS 50, Seq Length 80, Lr 1.5e-3, Batch size 6, model1')
    parser.add_argument("--experiment_name", type=str, default='train_eval3')

    #
    #
    parser.add_argument("--label_dist_fn", type=str, default='l1')
    parser.add_argument("--feat_dist_fn", type=str, default='max_corr')
    parser.add_argument("--label_temperature", type=float, default=1)
    # parser.add_argument("--label_temperature", type=float, default=0.1)

    opt = parser.parse_args()
    print(opt)
    if UMAP:
        train_data, train_label, train_freq = load_mnist(opt, train_dype='train', target_digit=3, data_size=2000)
    else:
        train_data, train_label, train_freq = load_mnist(opt, train_dype='train', data_size=10000)
        val_data, val_label, val_freq = load_mnist(opt, train_dype='val', data_size=100)

        # train_data, train_label, train_freq = load_mnist(opt, train_dype='train', data_size=5000, target_digit=3)
    DEBUG = False
    # dataset = p.CustomDataset(train_data, train_label, train_freq, opt)
    dataset = p.CustomDataset(train_data, train_label, train_freq, opt, debug=DEBUG)
    # data_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size)
    data_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                             pin_memory=True)
    if UMAP:
        pass
    else:
        val_dataset = p.CustomDataset(val_data, val_label, val_freq, opt, debug=DEBUG)
        val_data_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=opt.num_workers,
                                     pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = m.SimPer(opt)

    if UMAP:
        import umap
        param = torch.load('./params/model_0000.pth')
        model.load_state_dict(param)
        # model.to('cpu')

    model.to(device)
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
        all_labels, label_index = me.label_distance(all_speed1, all_speed2, opt.DEBUG,
                                       opt.label_dist_fn, opt.label_temperature)
        mini_batch_size = frames.shape[0]
        shape = frames.shape[2:]
        transform_shape = (mini_batch_size * num_arguments, *shape)
        frames_transformed = frames.view(transform_shape)
        if DEBUG:
            #
            # add channel dimension
            #
            frames_transformed = frames_transformed.unsqueeze(1)

        frames_transformed = frames_transformed.to(device)
        # frames_transformed = frames_transformed.to('cpu')
        all_z = model(frames_transformed, 'f')

        if UMAP:
            t.plot_umap(model, data_loader, device, opt)
            print('complete visualize')
            return
            # all_speed_transformed = all_speed.view((mini_batch_size * num_arguments, -1))
            # all_speed_arr = all_speed_transformed.detach().cpu().numpy().flatten()
            # data = all_z.detach().cpu().numpy()
            # embedding = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(data)
            # # plt.scatter(embedding[:, 0], embedding[:, 1], c=all_speed_arr, cmap='Spectral', s=5)
            # plt.scatter(embedding[:, 0], embedding[:, 1], c=all_speed_arr, cmap='Blues', s=5)
            # # plt.scatter(embedding[:, 0], embedding[:, 1])
            # plt.colorbar()
            # plt.show()

        all_z = all_z.view(mini_batch_size, num_arguments, -1)

        for feats, labels in zip(all_z, all_labels):
            labels = labels.to(device)
            # print(feats, labels)
            feat1 = feats[:half_of_num_arguments]
            feat2 = feats[half_of_num_arguments:]
            feat1_numpy = feat1.detach().cpu().numpy()
            feat2_numpy = feat2.detach().cpu().numpy()
            diff = feat1_numpy - feat2_numpy
            if opt.feat_dist_fn == 'max_corr':
                # feat_dist = me.max_cross_corr(feat1, feat2, device)
                feat_dist = me.batched_max_cross_corr(feat1, feat2, device)
                gen_infoNCE_loss1 = l.generalized_info_nce(feat_dist, labels, opt.DEBUG, temperature=1)
                gen_infoNCE_loss2 = l.generalized_info_nce(feat_dist, labels, opt.DEBUG, temperature=0.1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    t.train(data_loader, model, criterion, optimizer, device, opt, val_data_loader)

    print('complete')
    return


if __name__ == '__main__':
    main()
