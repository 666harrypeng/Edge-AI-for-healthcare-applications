import torch
import torch.nn as nn


##### LSTM Model Definition ##### 
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # shape(input tensors) should be (batch_size, sequence_length, input_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # cell state
        
        # forward pass through LSTM
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        out = self.fc(h_out[-1])
        return out