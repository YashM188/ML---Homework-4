# train_lenet1.py
import torch
from torch.utils.data import DataLoader
import torchvision

import mnist
from lenet5_rbf import LeNet5RBF   # from 1.1 (make sure lenet5_rbf.py is in same folder)


def map_loss(y, labels, j=0.1):
    """
    Implements Eq. (9) from the paper, with j = 0.1 and sum over incorrect classes.

    y:      [B,10] penalties y_k = ||h - mu_k||^2
    labels: [B]    correct digit indices (0..9)
    """
    device = y.device
    B = y.size(0)
    idx = torch.arange(B, device=device)

    # y_D: penalty of correct class
    y_correct = y[idx, labels]  # [B]

    # y_i for i != D
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[idx, labels] = False
    y_incorrect = y[mask].view(B, -1)  # [B, 9]

    # log( e^{-j} + sum_{i != D} e^{-y_i} )
    j_term = torch.exp(torch.tensor(-j, device=device))
    second_term = torch.log(j_term + torch.exp(-y_incorrect).sum(dim=1))

    # average over batch
    loss = (y_correct + second_term).mean()
    return loss


def evaluate_error(model, dataloader, device):
    """
    Compute classification error (1 - accuracy) using argmin over penalties.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            y = model(images)                 # [B,10]
            preds = torch.argmin(y, dim=1)    # smaller penalty => predicted class

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 1.0 - correct / total


def train_lenet1(num_epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pad 28x28 MNIST to 32x32 as in the paper
    pad = torchvision.transforms.Pad(2, fill=0, padding_mode='constant')

    train_dataset = mnist.MNIST(split="train", transform=pad)
    test_dataset  = mnist.MNIST(split="test",  transform=pad)

    # HW requires batch size = 1
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False)

    model = LeNet5RBF(mu_path="rbf_mu.pt").to(device)

    train_errors = []
    test_errors = []

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            y = model(images)
            loss = map_loss(y, labels, j=0.1)

            # Steepest gradient descent (no optimizer object)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p -= lr * p.grad

        # Evaluate train/test error at end of each epoch
        train_err = evaluate_error(model, train_loader, device)
        test_err  = evaluate_error(model, test_loader,  device)
        train_errors.append(train_err)
        test_errors.append(test_err)

        print(f"Epoch {epoch+1:02d} | train err: {train_err:.4f} | test err: {test_err:.4f}")

    # Save the trained model for Q1.3 (this answers your LeNet1.pth question)
    torch.save(model.state_dict(), "LeNet1.pth")

    # Optionally save error curves for 1.3 plots
    torch.save(
        {"train_errors": train_errors, "test_errors": test_errors},
        "lenet1_errors.pt"
    )

    print("Finished training. Saved model to LeNet1.pth")


if __name__ == "__main__":
    train_lenet1(num_epochs=20, lr=1e-3)
