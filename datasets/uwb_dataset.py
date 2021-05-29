import numpy as np
import torch
from torch.utils.data import Dataset
from utils import read_excel


class PosDataset(Dataset):
    def __init__(self, files, max_error):
        data = read_excel(files, concat=True, max_error=max_error)
        self.Y = torch.tensor(
            data[["reference__x", "reference__y"]].values.astype(np.float32)
        )
        self.X = torch.tensor(
            data[
                data.columns.difference(["reference__x", "reference__y"])
            ].values.astype(np.float32)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X[idx, :]
        y = self.Y[idx, :]

        return x, y
