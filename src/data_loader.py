import os
import numpy as np

def load_splits(split_path="splits/olivetti_splits.npz"):
    """
    Loads saved dataset splits from olivetti_splits.npz.
    Falls back to current directory if not found in splits/.
    """
    if not os.path.exists(split_path):
        if os.path.exists("olivetti_splits.npz"):
            split_path = "olivetti_splits.npz"
        else:
            raise FileNotFoundError(f"Split file not found at {split_path} or olivetti_splits.npz")

    data = np.load(split_path)
    splits = {
        "x_train": data["x_train"],
        "y_train": data["y_train"],
        "x_val": data["x_val"],
        "y_val": data["y_val"],
        "x_test": data["x_test"],
        "y_test": data["y_test"],
        "x_test_masked": data["x_test_masked"],
        "train_idx": data["train_idx"],
        "val_idx": data["val_idx"],
        "test_idx": data["test_idx"],
    }
    
    # Sanity checks
    assert splits["x_train"].shape == (240, 4096), f"Unexpected x_train shape: {splits['x_train'].shape}"
    assert splits["x_val"].shape == (80, 4096), f"Unexpected x_val shape: {splits['x_val'].shape}"
    assert splits["x_test"].shape == (80, 4096), f"Unexpected x_test shape: {splits['x_test'].shape}"
    assert len(np.intersect1d(splits["train_idx"], splits["val_idx"])) == 0, "Leakage between train and val!"
    assert len(np.intersect1d(splits["train_idx"], splits["test_idx"])) == 0, "Leakage between train and test!"
    assert len(np.intersect1d(splits["val_idx"], splits["test_idx"])) == 0, "Leakage between val and test!"
    
    return splits

def apply_center_mask(images, region=(24, 40, 24, 40)):
    """
    Applies center occlusion mask [r_min:r_max, c_min:c_max] to (N, 4096) images.
    """
    r_min, r_max, c_min, c_max = region
    masked = images.copy().reshape(-1, 64, 64)
    masked[:, r_min:r_max, c_min:c_max] = 0.0
    return masked.reshape(-1, 4096)
