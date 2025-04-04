import torch
import torch.nn as nn

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(IdentityBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels * 4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(out_channels * 4)

        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * 4, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(out_channels * 4)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)

        return out


class ResCNN(nn.Module):
    def __init__(self, num_classes=1, in_channels=1, out_channels=512):
        super(ResCNN, self).__init__()
        self.out_channel = out_channels
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(16, 16, 3)
        self.layer2 = self.make_layer(64, 32, 4, stride=2)
        self.layer3 = self.make_layer(128, 64, 6, stride=2)
        self.layer4 = self.make_layer(256, out_channels=out_channels, blocks=3, stride=2)

        self.out_conv = nn.Conv1d(in_channels=4*out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.out_conv_lon = nn.Conv1d(in_channels=4 * out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                  padding=1)
        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(2048, num_classes)

    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(IdentityBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(IdentityBlock(out_channels * 4, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, isOut = True, isLoc = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        if isOut:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        else:
            if isLoc:
                x_lat = self.out_conv(x)
                x_lat = self.avgpool(x_lat)
                x_lon = self.out_conv_lon(x)
                x_lon = self.avgpool(x_lon)
                return x_lat, x_lon
            else:
                x = self.out_conv(x)
                x = self.avgpool(x)
        return x
