# build_rbf_mu.py
import os
from glob import glob

import numpy as np
from PIL import Image
import torch


def build_digit_prototypes(digits_root="./digits updated",
                           out_path="rbf_mu.pt",
                           images_per_class=None):
    """
    Build 10 prototype vectors (one per digit 0..9) from the DIGITS dataset.

    digits_root: folder containing subfolders '0', '1', ..., '9' with digit images.
                 For example: "./digits updated/0/xxx.png", etc.
    out_path:    where to save the 10x84 tensor of RBF centers.
    images_per_class: optionally limit number of images per class (for speed).
    """
    mus = []

    for d in range(10):
        folder = os.path.join(digits_root, str(d))
        files = sorted(glob(os.path.join(folder, "*.png")))

        if len(files) == 0:
            raise RuntimeError(f"No PNG files found in {folder}")

        if images_per_class is not None:
            files = files[:images_per_class]

        acc = None
        count = 0

        # 1) Average all images for this digit
        for fn in files:
            img = Image.open(fn).convert("L")  # grayscale
            arr = np.asarray(img, dtype=np.float32)

            if acc is None:
                acc = np.zeros_like(arr, dtype=np.float32)

            acc += arr
            count += 1

        mean_img = acc / count  # e.g., 128x128 float image

        # 2) Normalize to [0,1]
        min_val = mean_img.min()
        max_val = mean_img.max()
        mean_img = (mean_img - min_val) / (max_val - min_val + 1e-8)

        # 3) Downsample to 7x12 (height x width)
        # PIL resize takes (width, height)
        small = Image.fromarray((mean_img * 255).astype(np.uint8)).resize(
            (12, 7), Image.BILINEAR
        )
        small_arr = np.asarray(small, dtype=np.float32) / 255.0  # shape (7,12)

        # 4) Binarize: > mean => 1, else 0
        thresh = small_arr.mean()
        bitmap = (small_arr > thresh).astype(np.float32)  # 0 or 1

        # 5) Map to {-1, +1} and flatten to length 84
        mu_vec = 2.0 * bitmap.reshape(-1) - 1.0  # 0 -> -1, 1 -> +1
        mus.append(mu_vec)

    mu = torch.tensor(np.stack(mus, axis=0), dtype=torch.float32)  # (10,84)
    torch.save(mu, out_path)
    print(f"Saved RBF centers to {out_path} with shape {mu.shape}")


if __name__ == "__main__":
    # Adjust digits_root if your DIGITS folder is named differently
    build_digit_prototypes(digits_root="./digits updated", out_path="rbf_mu.pt")
