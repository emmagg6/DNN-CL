# h_test.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from dataset import make_MNIST
from Models.Hnet.hnet_nn import hn, Hnet_train_epoch, Hnet_evaluate


def main():
    # hyperparams
    batch_size = 64
    lr = 1e-3
    epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load MNIST
    trainset, validset, testset = make_MNIST()
    train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=0, # slower but necessary due to loop of trials and datasets
                                            pin_memory=True,
                                            worker_init_fn=worker_init_fn)
    valid_loader = torch.utils.data.DataLoader(validset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            worker_init_fn=worker_init_fn)

    # instantiate model
    # no edge specs for full connectivity
    model = hn().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(1, epochs + 1):
        train_loss = Hnet_train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = Hnet_evaluate(model, valid_loader, criterion, device)
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {100*val_acc:5.2f}%")

    # final test
    test_loss, test_acc = Hnet_evaluate(model, test_loader, criterion, device)
    print(f"\nTest Set → Loss: {test_loss:.4f} | Acc: {100*test_acc:5.2f}%")

    # optionally save
    # torch.save(model.state_dict(), "mnist_hnet.pt")


if __name__ == "__main__":
    main()