import torch
import torch.nn as nn

class SoundModel(nn.Module):
    def __init__(self, num_classes=5):
        super(SoundModel, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.dropout = nn.Dropout(0.4)        

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(166400, 64)
        self.fc2 = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x   