# q1_3_eval_fixed.py
import os
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torchvision
import mnist
from lenet5_rbf import LeNet5RBF
import matplotlib.pyplot as plt


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same test set / padding as training
    pad = torchvision.transforms.Pad(2, fill=0, padding_mode='constant')
    test_dataset = mnist.MNIST(split="test", transform=pad)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Rebuild model and load weights
    state_dict = torch.load("LeNet1.pth", map_location=device)
    model = LeNet5RBF(mu_path="rbf_mu.pt").to(device)
    model.load_state_dict(state_dict)
    model.eval()

    num_classes = 10
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    # FIXED: Track smallest penalty (most confident) misclassification
    best_penalty = [float('inf')] * num_classes  # Start with infinity
    best_example = [None] * num_classes

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)               # [1,10] penalties
            preds = torch.argmin(outputs, dim=1)  # [1]

            true = labels.item()
            pred = preds.item()

            # Fill confusion matrix
            confusion[true, pred] += 1

            # FIXED: Track most confident wrong prediction (smallest penalty)
            if true != pred:
                penalty = outputs[0, pred].item()
                if penalty < best_penalty[true]:  # Smaller penalty = more confident
                    best_penalty[true] = penalty
                    best_example[true] = (idx, true, pred, images[0].cpu())

    # Print summary
    print("\nMisclassification Summary:")
    print("-" * 60)
    for d in range(num_classes):
        total = confusion[d].sum().item()
        correct = confusion[d, d].item()
        errors = total - correct
        print(f"Digit {d}: {errors} misclassifications out of {total} samples")
    print("-" * 60)

    # Save most confusing examples
    os.makedirs("most_confusing", exist_ok=True)
    imgs_for_grid = []
    titles_for_grid = []

    print("\nMost Confusing Examples (smallest penalty = highest confidence):")
    print("-" * 60)
    for d in range(num_classes):
        info = best_example[d]
        if info is None:
            print(f"Digit {d}: No misclassified examples (100% accuracy!)")
            continue

        idx, true, pred, img = info
        penalty = best_penalty[d]
        print(f"Digit {d}: index {idx}, predicted as {pred}, penalty={penalty:.4f}")

        arr = img.numpy().squeeze()
        pil_img = Image.fromarray(arr.astype(np.uint8))
        filename = f"most_confusing/digit_{true}_pred_{pred}_idx_{idx}.png"
        pil_img.save(filename)

        imgs_for_grid.append(arr)
        titles_for_grid.append(f"{true}→{pred} (idx {idx})")

    print("-" * 60)

    # Show grid
    if len(imgs_for_grid) > 0:
        n = len(imgs_for_grid)
        cols = min(5, n)
        rows = math.ceil(n / cols)

        fig2, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
        axes = np.array(axes).reshape(-1)

        for k, (img_arr, title) in enumerate(zip(imgs_for_grid, titles_for_grid)):
            ax = axes[k]
            ax.imshow(img_arr, cmap="gray")
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        for k in range(len(imgs_for_grid), len(axes)):
            axes[k].axis("off")

        fig2.suptitle("Most Confusing Misclassified Example per Digit", fontsize=12)
        plt.tight_layout()
        fig2.savefig("most_confusing_grid.png", dpi=200, bbox_inches="tight")

    # Save confusion matrix
    torch.save(confusion, "lenet1_confusion.pt")

    # Plot confusion matrix
    cm = confusion.numpy()
    classes = list(range(10))

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=cm.max())
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel("Count")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig("lenet1_confusion_matrix_blue.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()