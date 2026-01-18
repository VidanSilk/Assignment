# model.py
import torch
import torch.nn as nn
from .baseBlocks import *

class SpeechEnhancementLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SpeechEnhancementLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Add initial convolutional layers
        self.conv1 = convLayer(1, 16, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = convLayer(16, 32, kernel_size=(3, 3), stride=(1, 1))
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size * 32, hidden_size, num_layers, batch_first=True)

        # Fully connected layers
        self.fc1 = fcLayer(hidden_size, 256)
        self.fc2 = fcLayer(256, output_size)
    
    def forward(self, x):
        # Apply initial convolutional layers
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.conv2(x)

        # Reshape for LSTM
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, time, -1)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Fully connected layers
        out = self.fc1(out)
        out = self.fc2(out)
        return out