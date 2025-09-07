from torch.utils.data import Dataset
import torch
import pandas as pd

class FriendshipCSVDataset(Dataset):
    def __init__(self, x_data_path, labels_data_path):
        x_df = pd.read_csv(x_data_path)
        y_df = pd.read_csv(labels_data_path)

        # Ensure consistent column names (changes in-memory dataframe only)
        # x_df = x_df.rename(columns={x_df.columns[0]: "subj_i", x_df.columns[1]: "subj_b"})
        # y_df = y_df.rename(columns={y_df.columns[0]: "subj_j", y_df.columns[1]: "subj_b"})

        # Inner join on (subj_a, subj_b)
        df = pd.merge(x_df, y_df, on=["subj_i", "subj_j"], how="inner")

        print(f"Loaded {len(df)} samples after join.")

        # Features = all columns from x_df except subj_a, subj_b
        features = df.iloc[:, 2: -1]

        # Label = last column from y_df (after join)
        labels = df.iloc[:, -1]

        self.X = features.values.astype("float32")
        self.y = labels.values.astype("int64")

        print(f"Feature shape: {self.X.shape}, Labels shape: {self.y.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])  # shape: (num_features,), # ROI features
        if x.ndim == 1:
            x = x.unsqueeze(0)  # shape: (1, num_features) for Conv1D
        y = torch.tensor(self.y[idx], dtype=torch.long)  # shape: scalar of class label
        return x, y