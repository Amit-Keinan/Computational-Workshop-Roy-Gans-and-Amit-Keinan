import torch
import torch.nn as nn
import torch.nn.functional as F

class Friendship1DCNN(nn.Module):
    def __init__(self, num_classes, num_input_features):
        super(Friendship1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        # Compute length after pooling
        pooled_length = num_input_features // 2  # only one pooling layer
        self.fc1 = nn.Linear(32 * pooled_length, 64)
        self.fc2 = nn.Linear(64, num_classes)

        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # (batch, 32, 45)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # logits for softmax or ordinal head
