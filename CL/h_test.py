# h_test.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from dataset import make_MNIST
from hnet.hnet_nn import hn


def main():
    # hyperparams
    batch_size = 64
    lr = 1e-3
    epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load MNIST
    train_set, valid_set, test_set = make_MNIST()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2)

    # instantiate model
    # you can tweak n_random_edges per layer or leave None for full connectivity
    model = hn(n_edges1=500, n_edges2=1000).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {100*val_acc:5.2f}%")

    # final test
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Set → Loss: {test_loss:.4f} | Acc: {100*test_acc:5.2f}%")

    # optionally save
    torch.save(model.state_dict(), "mnist_hnet.pt")


if __name__ == "__main__":
    main()