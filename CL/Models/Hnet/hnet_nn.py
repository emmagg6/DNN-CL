# hnet_nn.py
import torch
import torch.nn as nn
import torch.optim as optim

from hnetconv2d import HNetConv2d  


class hn(nn.Module):
    def __init__(self, n_edges1=None, n_edges2=None):
        super().__init__()
        # first HNet conv: 1→16 channels, 3×3, padding=1
        self.conv1 = HNetConv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            n_random_edges=n_edges1,
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)  # 28→14

        # second HNet conv: 16→32 channels, 3×3, padding=1
        self.conv2 = HNetConv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            n_random_edges=n_edges2,
        )
        # after pool: 14→7

        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # x: (B, 784)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return avg_loss, acc