import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Convolutional Neural Network for the classification
    of dermoscopic images into seven diagnostic classes.
    """

    def __init__(self):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(
            3, 48, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(48)

        self.conv2 = nn.Conv2d(
            48, 48, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(48)

        # Block 2
        self.conv3 = nn.Conv2d(
            48, 96, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(
            96, 96, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(96)

        # Block 3
        self.conv5 = nn.Conv2d(
            96, 192, kernel_size=3, padding=1
        )
        self.bn5 = nn.BatchNorm2d(192)

        self.conv6 = nn.Conv2d(
            192, 192, kernel_size=3, padding=1
        )
        self.bn6 = nn.BatchNorm2d(192)

        # Block 4
        self.conv7 = nn.Conv2d(
            192, 384, kernel_size=3, padding=1
        )
        self.bn7 = nn.BatchNorm2d(384)

        self.conv8 = nn.Conv2d(
            384, 384, kernel_size=3, padding=1
        )
        self.bn8 = nn.BatchNorm2d(384)

        # 2 x 2 max pooling
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classification layer
        self.fc = nn.Linear(384, 7)

    def forward(self, x):

        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)

        # Block 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool(x)

        # Global Average Pooling
        x = self.gap(x)

        # [batch, 384, 1, 1] -> [batch, 384]
        x = torch.flatten(x, 1)

        # Seven output logits
        x = self.fc(x)

        return x