import torch
import torch.nn as nn
import torch.nn.functional as F


class Featurizer(nn.Module):
    def __init__(self):
        super(Featurizer, self).__init__()
        self.conv0 = nn.Conv3d(1, 64, (5, 3, 3), padding=(2, 1, 1))
        self.conv1 = nn.Conv3d(64, 128, (5, 3, 3), padding=(2, 1, 1))
        self.conv2 = nn.Conv3d(128, 128, (5, 3, 3), padding=(2, 1, 1))
        self.conv3 = nn.Conv3d(128, 1, (1, 1, 1))

        self.bn0 = nn.BatchNorm3d(64)
        self.bn1 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(1)

        self.pool0 = nn.MaxPool3d((1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool0(F.relu(self.bn0(self.conv0(x))))
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # x = self.pool2(F.relu(self.bn2(self.conv2(x)))) # commented out as in original
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        return x


if __name__ == '__main__':
    batch_size = 16
    sequence = 5
    img_size = 16
    channels = 1
    shape = (8, 1, 10, 20, 20)
    x = torch.randn(shape)
    # 8 is the batch size here, and the input has a depth of 10, height and width of 20.
    model = Featurizer()
    out = model(x)
    print(out)