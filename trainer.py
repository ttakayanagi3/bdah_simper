import torch
import torch.nn as nn


def trainer(data_loader, model, optimizer, opt):
    for i in range(opt.n_epochs):
        for x, y_label, y_angle in data_loader:
            out = model(x)

