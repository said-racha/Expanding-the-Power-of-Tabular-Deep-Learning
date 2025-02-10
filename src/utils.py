import numpy as np
import torch


def initialize_device():
    """
    Initialize the device (CPU, GPU, or MPS)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


def dataloader_to_numpy(dataloader):
    """
    Convert a PyTorch DataLoader to numpy arrays.
    """
    X_list, y_list = [], []

    for X, y in dataloader:
        X_list.append(X.numpy())
        y_list.append(y.numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y
