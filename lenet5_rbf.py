# lenet5_rbf.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5RBF(nn.Module):
    """
    LeNet-5-like architecture with an RBF output layer.

    Input : [B, 1, 32, 32] grayscale image (MNIST padded to 32x32).
    Output: [B, 10] penalty values y_k = ||h - mu_k||^2 (smaller is better).
    """

    def __init__(self, mu_path="rbf_mu.pt"):
        super().__init__()

        # C1: 1x32x32 -> 6x28x28 (5x5 conv, stride 1)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)

        # S2: 6x28x28 -> 6x14x14 (2x2 avg pool)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C3: 6x14x14 -> 16x10x10 (5x5 conv, stride 1)
        # (We approximate the original partial connection scheme with a full 6->16 conv.)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)

        # S4: 16x10x10 -> 16x5x5 (2x2 avg pool)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C5: 16x5x5 -> 120x1x1 (5x5 conv)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)

        # F6: 120 -> 84 fully connected
        self.fc1 = nn.Linear(120, 84)

        # Load 10x84 RBF center vectors mu_k, treated as fixed (non-trainable) parameters.
        mu = torch.load(mu_path)  # shape should be (10, 84)
        assert mu.shape == (10, 84), f"Expected (10,84) mu, got {mu.shape}"
        self.register_buffer("mu", mu)

    def forward(self, x):
        """
        x: [B, 1, 32, 32]
        returns: [B, 10] penalties y_k
        """
        # C1 + tanh
        x = torch.tanh(self.conv1(x))   # -> [B, 6, 28, 28]

        # S2
        x = self.pool1(x)               # -> [B, 6, 14, 14]

        # C3 + tanh
        x = torch.tanh(self.conv2(x))   # -> [B, 16, 10, 10]

        # S4
        x = self.pool2(x)               # -> [B, 16, 5, 5]

        # C5 + tanh
        x = torch.tanh(self.conv3(x))   # -> [B, 120, 1, 1]

        # Flatten
        x = x.view(x.size(0), -1)       # -> [B, 120]

        # F6 + tanh => h in R^{B x 84}
        h = torch.tanh(self.fc1(x))     # -> [B, 84]

        # RBF output: y_k = ||h - mu_k||^2
        diff = h.unsqueeze(1) - self.mu.unsqueeze(0)   # [B, 10, 84]
        y = (diff ** 2).sum(dim=2)                     # [B, 10]

        return y
