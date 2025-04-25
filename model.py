import torch 
import torch.nn as nn 
 
class SiameseCNN1D(nn.Module): 
    def __init__(self, input_channels=84, embedding_size=128): 
        super(SiameseCNN1D, self).__init__() 
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) 
        self.fc1 = nn.Linear(64 * 15, embedding_size) 
        self.relu = nn.ReLU() 
 
    def forward(self, x): 
        x = x.transpose(1, 2) 
        x = self.relu(self.conv1(x)) 
        x = self.pool(x) 
        x = x.view(x.size(0), -1) 
        embedding = self.fc1(x) 
        return embedding 
