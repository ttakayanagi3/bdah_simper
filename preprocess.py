import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader


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

    # angles = [90.0 * np.sin(2 * np.pi * y['freq'] * idx / opt.FPS + offset) for idx in range(NUM_FRAMES)]
    angles = [90.0 * np.sin(2 * np.pi * y.numpy() * idx / opt.FPS + offset) for idx in range(NUM_FRAMES)]

    new_x = []
    for i in range(NUM_FRAMES):
        x_ = x.unsqueeze(0)
        x_rotate = F.rotate(x_, angles[i] * np.pi / 180.0)
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

    arbitrary_speed_subsample(batched_frames, random_speeds, opt)

    return random_speeds


def arbitrary_speed_subsample(frames, speed, opt):
    frame_len = frames.shape[0]
    # frame_len = torch.FloatTensor(frame_len * )
    max_frame_len = torch.floor_divide(frame_len, speed)

    return


def transform_simper(frames, opt):
    num_diff_speeds = opt.NUM_SELF_CON_SIMPER
    speed_range = (0.5, opt.MAX_SPEED)

    temp = batched_arbitrary_speed(frames, num_diff_speeds, speed_range, opt)



    pass



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
        _ = transform_simper(X, self.opt)

        return X, y_label, y_angle


