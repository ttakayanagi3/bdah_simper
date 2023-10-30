import torch
import torch.nn as nn
import torch.nn.functional as F


class Featurizer(nn.Module):
    def __init__(self, n_outputs):
        super(Featurizer, self).__init__()
        self.conv0 = nn.Conv3d(1, 64, (5, 3, 3), padding=(2, 1, 1))
        self.conv1 = nn.Conv3d(64, 128, (5, 3, 3), padding=(2, 1, 1))
        self.conv2 = nn.Conv3d(128, 128, (5, 3, 3), padding=(2, 1, 1))
        self.conv3 = nn.Conv3d(128, 1, (1, 1, 1))

        self.bn0 = nn.BatchNorm3d(64)
        self.bn1 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(1)

        # self.pool0 = nn.MaxPool3d((1, 2, 2))
        # self.pool1 = nn.MaxPool3d((1, 2, 2))
        # self.pool2 = nn.MaxPool3d((1, 2, 2))
        # self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.pool0 = nn.MaxPool2d((2, 2))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()

        self.n_outputs = n_outputs

    # def forward(self, x):
    #     x = self.pool0(F.relu(self.bn0(self.conv0(x))))
    #     x = self.pool1(F.relu(self.bn1(self.conv1(x))))
    #     # x = self.pool2(F.relu(self.bn2(self.conv2(x)))) # commented out as in original
    #     x = self.pool3(F.relu(self.bn3(self.conv3(x))))
    #     x = self.flatten(x)
    #     return x

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.time_distributed(x, self.pool0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.time_distributed(x, self.pool1)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.time_distributed(x, self.pool3)

        x = self.flatten(x)
        return x

    def time_distributed(self, x, layer):
        channels = x.shape[1]
        sequence = x.shape[2]
        img_size = x.shape[3:]

        new_order = (0, 2, 1, 3, 4)  # Batch, Sequence, Channel, Height, Width
        x_trans = x.permute(new_order)
        new_shape = (channels, img_size[0], img_size[1])
        x_trans = x_trans.reshape(-1, *new_shape)  # Batch x Sequence, Channel, Height, Width
        out = layer(x_trans)
        img_size = out.shape[2:]
        out = out.reshape(-1, sequence, channels, img_size[0], img_size[1])
        out = out.permute(0, 2, 1, 3, 4)  # Batch, Channel, Sequence, Height, Width
        return out


class MLP(nn.Module):
    def __init__(self, n_outputs):
        super(MLP, self).__init__()
        self.inputs = nn.Linear(n_outputs, n_outputs)
        self.hidden = nn.Linear(n_outputs, n_outputs)
        self.outputs = nn.Linear(n_outputs, n_outputs)

    def forward(self, x):
        x = F.relu(self.inputs(x))
        # x = F.relu(self.hidden(x)) # commented out as in original
        x = self.outputs(x)
        return x


def Classifier(in_features, out_features, nonlinear=False):
    if nonlinear:
        return nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, out_features)
        )
    else:
        return nn.Linear(in_features, out_features)


class SimPer(nn.Module):
    def __init__(self, opt):
        super(SimPer, self).__init__()
        self.featurizer = Featurizer(opt.SSL_FRAMES)
        self.reg = Classifier(1, 1, False)
        self.network = nn.Sequential(*[self.featurizer, self.reg])

    def forward(self, x, inference='f'):
        if inference == 'f':
            out = self.featurizer(x)
        else:
            out = self.network(x)
        return out


if __name__ == '__main__':
    batch_size = 16
    sequence = 5
    img_size = 16
    channels = 1
    shape = (batch_size, channels, sequence, img_size, img_size)
    x = torch.randn(shape)
    # 8 is the batch size here, and the input has a depth of 10, height and width of 20.
    model = Featurizer(n_outputs=50)
    out = model(x)
    print(out)