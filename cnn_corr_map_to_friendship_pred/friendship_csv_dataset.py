from torch.utils.data import Dataset
import torch
import pandas as pd

class FriendshipCSVDataset(Dataset):
    def __init__(self, csv_path, x_col_name, y_col_name):
        df = pd.read_csv(csv_path)
        
        # Assume last column is label, others are features
        self.X = df[x_col_name].values.astype("float32")
        self.y = df[y_col_name].values.astype("int64")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])  # shape: (num_features,)
        if x.ndim == 1:
            x = x.unsqueeze(0)  # shape: (1, num_features) for Conv1D
        y = torch.tensor(self.y[idx], dtype=torch.long)  # shape: scalar of class label
        return x, y