import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader
from friendship_csv_dataset import FriendshipCSVDataset

class FriendshipCSVDataLoader():
    def __init__(self, x_csv_path, y_csv_path):
       self.dataset = FriendshipCSVDataset(x_csv_path, y_csv_path)

    def get_data_loaders(self, batch_size, train_test_split):
        split_idx = int(len(self.dataset) * train_test_split)

        train_dataset = torch.utils.data.Subset(self.dataset, range(0, split_idx))
        test_dataset  = torch.utils.data.Subset(self.dataset, range(split_idx, len(self.dataset)))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

