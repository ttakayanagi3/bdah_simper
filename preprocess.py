import argparse
import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.functional as F2
from torchvision import transforms as tr

from torch.utils.data import Dataset, DataLoader
# from torchinterp1d import interp1d

import main as m

def rescale(x, scaled_value=255.0):
    return x / scaled_value


def clip_value(x):
    x = torch.clamp(x, min=0, max=1)
    return x


def gen_rotating_samples(x, y, opt):
    if opt.random_phase == 1:
        offset = np.random.uniform(low=-1.57, high=1.57)
    else:
        offset = 0

    NUM_FRAMES = opt.FPS * opt.LENGTH_SEC

    angles = [90.0 * np.sin(2 * np.pi * y.numpy() * idx / opt.FPS + offset) for idx in range(NUM_FRAMES)]

    new_x = []
    for i in range(NUM_FRAMES):
        x_ = x.unsqueeze(0)
        # x_rotate = F.rotate(x_, angles[i] * np.pi / 180.0)
        x_rotate = F.rotate(x_, angles[i])
        x_rotate = x_rotate.numpy()
        new_x.append(x_rotate)
    new_x_arr = np.array(new_x)
    y_angle = torch.FloatTensor(angles) * np.pi / 180.0
    new_x_tensor = torch.FloatTensor(new_x_arr).squeeze()

    return new_x_tensor, y_angle


def preprocessing(x, y_freq, opt):
    x = rescale(x)
    x, y_angle = gen_rotating_samples(x, y_freq, opt)
    x = clip_value(x)
    return x, y_angle


def batched_arbitrary_speed(frames, num_diff_speeds, speed_range, opt):
    random_speeds = np.random.uniform(low=speed_range[0], high=speed_range[1], size=num_diff_speeds)
    random_speeds = np.sort(random_speeds)
    random_speeds = np.concatenate((random_speeds, random_speeds))

    batched_frames = torch.stack([frames] * num_diff_speeds * 2)

    batched_adjusted_frames = arbitrary_speed_subsample(batched_frames, random_speeds, opt)

    return batched_adjusted_frames, random_speeds


def arbitrary_speed_subsample(frames, speed, opt):
    frame_len = frames.shape[1]
    interp_frames = []
    # extract_time_frames_length = 100
    # extract_time_frames_length = 60
    extract_time_frames_length = opt.extract_time_frames
    for i, each_speed in enumerate(speed):
        frame = frames[i].view(frame_len, -1).T.unsqueeze(0)
        target_frame_len = int(frame_len // each_speed)
        target_output_size = target_frame_len
        interp_frame = F2.interpolate(frame, size=target_output_size, mode='linear')
        if opt.DEBUG == 1:
            frame_numpy = frame.numpy()
            interp_frame_numpy = interp_frame.squeeze().numpy()
            interp_frame_numpy = interp_frame_numpy.T.reshape(-1, 28, 28)
        #
        # reshape (784, adjusted time frames) -> (adjusted time frame, 28, 28)
        #
        interp_frame = interp_frame.squeeze().T.reshape(-1, opt.IMG_SIZE, opt.IMG_SIZE)
        #
        # extract first 50 frames
        #
        interp_frames.append(interp_frame[:extract_time_frames_length, :, :])
    interp_frames_t = torch.stack(interp_frames)
    return interp_frames_t


def transform_simper(frames, opt, debug=False):
    num_diff_speeds = opt.NUM_SELF_CON_SIMPER
    speed_range = (0.5, opt.MAX_SPEED)
    #
    # temporal variant augmentation
    #
    different_speed_batched_frames, random_speeds = batched_arbitrary_speed(frames, num_diff_speeds, speed_range, opt)
    if debug:
        return different_speed_batched_frames, random_speeds
    #
    # temporal invariant augmentation
    #
    random_crop = tr.RandomResizedCrop(size=(28, 28))
    random_jitter = tr.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
    random_horizontal_flip = tr.RandomHorizontalFlip(p=1)
    gaussian_blur = tr.GaussianBlur(kernel_size=(3, 3))
    batched_transformed_frames = []
    for batched_frames in different_speed_batched_frames:
        p_random_crop = random.random()
        p_jitter = random.random()
        p_horizontal_flip = random.random()
        p_blur = random.random()
        transformed_frames = []
        for frame in batched_frames:
            #
            # need to add Channel dimension
            #
            frame = frame.unsqueeze(0)
            #
            # apply temporal variant transformation randomly
            #
            # if p_random_crop > 0.5:
            #     frame = random_crop(frame)
            # if p_jitter > 0.5:
            #     frame = random_jitter(frame)
            # if p_horizontal_flip > 0.5:
            #     frame = random_horizontal_flip(frame)
            if p_blur > 0.5:
                frame = gaussian_blur(frame)

            transformed_frames.append(frame)
        batched_transformed_frames.append(torch.stack(transformed_frames))
    batched_transformed_frames_t = torch.stack(batched_transformed_frames)
    #
    # insert Channel dimension
    #
    batched_transformed_frames_t = batched_transformed_frames_t.permute(0, 2, 1, 3, 4)

    return batched_transformed_frames_t, random_speeds


class CustomDataset(Dataset):
    def __init__(self, X, y_label, y_freq, opt,
                 need_preprocess=True, need_transform=True, debug=False, invariant_transform=True):
        self.X = X
        self.y_label = y_label
        self.y_freq = y_freq
        self.opt = opt
        self.need_preprocess = need_preprocess
        self.need_transform = need_transform
        self.debug = debug
        self.invariant_transform = invariant_transform
        print(f'debug_mode: {self.debug}')

    def __len__(self):
        return len(self.y_label)

    def __getitem__(self, idx):
        X = self.X[idx]
        y_label = self.y_label[idx]
        y_freq = self.y_freq[idx]
        #
        # rescale & rotation & clip
        #
        if self.need_preprocess:
            X, y_angle = preprocessing(X, y_freq, self.opt)
        else:
            # dummy
            y_angle = np.array([1])
        #
        # transform
        #
        if self.need_transform:
            # X, speed = transform_simper(X, self.opt)
            X, speed = transform_simper(X, self.opt, debug=self.debug)
        else:
            # dummy
            speed = np.array([1])

        return X, speed, y_angle

if __name__ == '__main__':
    FPS = 50
    LENGTH_SEC = 5
    NUM_FRAMES = FPS * LENGTH_SEC
    MAX_SPEED = 3
    IMG_SIZE = 28
    CHANNELS = 1
    DEBUG = True

    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_freq", type=float, default=0.5, help="minimum frequency")
    parser.add_argument("--max_freq", type=float, default=5, help="minimum frequency")
    parser.add_argument("--random_phase", type=int, default=1, help="1: apply random phase, 0: default")
    parser.add_argument("--FPS", type=int, default=FPS)
    parser.add_argument("--LENGTH_SEC", type=int, default=LENGTH_SEC)
    # parser.add_argument("--batch_size", type=int, default=4, help="size of mini-batches")
    parser.add_argument("--batch_size", type=int, default=2, help="size of mini-batches")
    # parser.add_argument("--batch_size", type=int, default=8, help="size of mini-batches")
    parser.add_argument("--n_epochs", type=int, default=15, help="epochs")
    parser.add_argument("--NUM_SELF_CON_SIMPER", type=int, default=10)
    parser.add_argument("--MAX_SPEED", type=int, default=MAX_SPEED)
    parser.add_argument("--SSL_FRAMES", type=int, default=NUM_FRAMES // MAX_SPEED)
    parser.add_argument("--IMG_SIZE", type=int, default=IMG_SIZE)
    parser.add_argument("--CHANNELS", type=int, default=CHANNELS)
    parser.add_argument("--extract_time_frames", type=int, default=80)
    # parser.add_argument("--lr", type=float, default=2e-3, help="Learning Rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--DEBUG", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    # parser.add_argument("--experiment_name", type=str, default='FPS 50, Seq Length 80, Lr 1e-3, Batch size 4')
    parser.add_argument("--experiment_name", type=str, default='FPS 50, Seq Length 80, Lr 1e-3, Batch size 2, model2')
    parser.add_argument("--label_dist_fn", type=str, default='l1')
    parser.add_argument("--feat_dist_fn", type=str, default='max_corr')
    parser.add_argument("--label_temperature", type=float, default=1)
    # parser.add_argument("--label_temperature", type=float, default=0.1)

    opt = parser.parse_args()
    print(opt)
    # print('temp')
    train_data, train_label, train_freq = m.load_mnist(opt, train_dype='train')

    # dataset = CustomDataset(train_data, train_label, train_freq, opt,
    #                         need_preprocess=True, need_transform=True, debug=True)
    dataset = CustomDataset(train_data, train_label, train_freq, opt,
                            need_preprocess=True, need_transform=True, debug=DEBUG)
    data_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size)

    data_iter = iter(data_loader)
    frames, all_speed, y_angle = next(data_iter)
    frame_sampled = frames[0]
    num_speeds = frame_sampled.shape[0] // 2
    frame_sampled_1 = frame_sampled[num_speeds:]
    frame_sampled_2 = frame_sampled[:num_speeds]
    diff = frame_sampled_1 - frame_sampled_2
    diff_numpy = diff.detach().cpu().numpy()

    if DEBUG:
        pass
    else:
        frames = frames.squeeze()

    num_frames = 60
    num_speed = 5
    j = 1
    for i in range(num_frames):
        fig = plt.figure(figsize=(4, 4))
        fig = plt.title(j)
        fig = sns.heatmap(frames[0, j, i, :, :])
        plt.show()
    print('complete')
    # for j in range(num_speed):
    #     for i in range(num_frames):
    #         fig = plt.figure(figsize=(4, 4))
    #         fig = plt.title(j)
    #         fig = sns.heatmap(frames[0, j, i, :, :])
    #         plt.show()