import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.functional as F2
from torchvision import transforms as tr

from torch.utils.data import Dataset, DataLoader
# from torchinterp1d import interp1d


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

    y_angle = torch.FloatTensor(angles) * np.pi / 180.0
    new_x_tensor = torch.FloatTensor(new_x).squeeze()

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
    extract_time_frames_length = 50
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


def transform_simper(frames, opt):
    num_diff_speeds = opt.NUM_SELF_CON_SIMPER
    speed_range = (0.5, opt.MAX_SPEED)
    #
    # temporal variant augmentation
    #
    different_speed_batched_frames, random_speeds = batched_arbitrary_speed(frames, num_diff_speeds, speed_range, opt)
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
            frame_ = frame.unsqueeze(0)
            #
            # apply temporal variant transformation randomly
            #
            if p_random_crop > 0.5:
                frame_ = random_crop(frame_)
            if p_jitter > 0.5:
                frame_ = random_jitter(frame_)
            if p_horizontal_flip > 0.5:
                frame_ = random_horizontal_flip(frame_)
            if p_blur > 0.5:
                frame_ = gaussian_blur(frame_)

            transformed_frames.append(frame_)
        batched_transformed_frames.append(torch.stack(transformed_frames))
    batched_transformed_frames_t = torch.stack(batched_transformed_frames)
    return batched_transformed_frames_t, random_speeds


class CustomDataset(Dataset):
    def __init__(self, X, y_label, y_freq, opt):
        self.X = X
        self.y_label = y_label
        self.y_freq = y_freq
        self.opt = opt

    def __len__(self):
        return len(self.y_label)

    def __getitem__(self, idx):
        X = self.X[idx]
        y_label = self.y_label[idx]
        y_freq = self.y_freq[idx]
        #
        # rescale & rotation & clip
        #
        X, y_angle = preprocessing(X, y_freq, self.opt)
        #
        # transform
        #
        X, speed = transform_simper(X, self.opt)

        return X, speed, y_angle


