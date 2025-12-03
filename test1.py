from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import mnist
import torch
import numpy as np
import torchvision

 
def test(dataloader,model):

    #please implement your test code#
    ##HERE##
    import importlib   # local import, allowed inside the function
    import torch

    # 'model' here is actually a state_dict, not a full model object
    lenet_module = importlib.import_module("lenet5_rbf")
    LeNet5RBF = getattr(lenet_module, "LeNet5RBF")

    net = LeNet5RBF(mu_path="rbf_mu.pt")
    net.load_state_dict(model)   # load the weights
    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            # Forward pass: outputs are penalties y_k = ||h - mu_k||^2
            outputs = net(images)              # shape [B, 10]

            # Smaller penalty => more confident ⇒ use argmin
            preds = torch.argmin(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total
    ###########################                                                                                                                                                                            

    print("test accuracy:", test_accuracy)

 

def main():

    pad=torchvision.transforms.Pad(2,fill=0,padding_mode='constant')

    mnist_test=mnist.MNIST(split="test",transform=pad)

    test_dataloader= DataLoader(mnist_test,batch_size=1,shuffle=False)

    model = torch.load("LeNet1.pth")

    test(test_dataloader,model)

 

if __name__=="__main__":

    main()
