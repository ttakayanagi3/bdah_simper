import os
import argparse

import torch
from torchvision import datasets as ds
from torch.utils.data import Dataset, DataLoader

import preprocess as p

# class CustomDataset(Dataset):
#     def __init__(self):



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

    parser = argparse.ArgumentParser()
    parser.add_argument("--min_freq", type=float, default=0.5, help="minimum frequency")
    parser.add_argument("--max_freq", type=float, default=5, help="minimum frequency")
    parser.add_argument("--random_phase", type=int, default=1, help="1: apply random phase, 0: default")
    parser.add_argument("--FPS", type=int, default=FPS)
    parser.add_argument("--LENGTH_SEC", type=int, default=LENGTH_SEC)
    parser.add_argument("--batch_size", type=int, default=32, help="size of mini-batches")
    parser.add_argument("--NUM_SELF_CON_SIMPER", type=int, default=10)
    parser.add_argument("--MAX_SPEED", type=int, default=MAX_SPEED)
    parser.add_argument("--SSL_FRAMES", type=int, default=NUM_FRAMES // MAX_SPEED)


    # parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    # parser.add_argument("--b1", type=float, default=0.5, help="hyper parameter of adam")
    # parser.add_argument("--b2", type=float, default=0.999, help="hyper parameter of adam")
    # parser.add_argument("--in_channels", type=int, default=3, help="number of in channels")
    # parser.add_argument("--param_file", type=str, default="./params/Generator_108.pth", help="parameter file for load")
    # parser.add_argument("--phase", type=str, default="test", help="phase of model")
    # parser.add_argument("--eval_epochs", type=int, default=10, help="evaluation period")
    # parser.add_argument("--model_name", type=str, default="simple-cnn", help="model name for tensor board file")
    # parser.add_argument("--file_name", type=str, default="", help="file name to read")
    # parser.add_argument("--out_dir", type=str, default="./learning_process", help="directory name for output")
    opt = parser.parse_args()
    print(opt)


    train_data, train_label, train_freq = load_mnist(opt, train_dype='train')
    dataset = p.CustomDataset(train_data, train_label, train_freq, opt)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size)
    data_iter = iter(dataloader)
    x, y_label, y_angle = next(data_iter)
    # train_data = p.preprocessing(train_data, train_freq, opt)
    print('complete')
    return
#


if __name__ == '__main__':
    main()
