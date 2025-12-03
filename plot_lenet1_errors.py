# plot_lenet1_errors.py
import torch
import matplotlib.pyplot as plt

def main():
    data = torch.load("lenet1_errors.pt")
    train_errors = data["train_errors"]
    test_errors = data["test_errors"]

    epochs = range(1, len(train_errors) + 1)

    plt.figure()
    plt.plot(epochs, train_errors, label="Train error")
    plt.plot(epochs, test_errors, label="Test error")
    plt.xlabel("Epoch")
    plt.ylabel("Error rate")
    plt.title("LeNet-1 training and test error")
    plt.grid(True)
    plt.legend()
    plt.savefig("lenet1_error_curve.png")  # include this in your PDF
    plt.show()

if __name__ == "__main__":
    main()
