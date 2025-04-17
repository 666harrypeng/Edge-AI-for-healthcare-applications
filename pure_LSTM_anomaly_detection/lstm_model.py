import torch
import torch.nn as nn


##### LSTM Model Definition ##### 
class LSTMModel(nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.l_win = config['l_win']
        self.num_hidden_units_lstm = config['num_hidden_units_lstm']
        self.input_size = config['input_size'] 
        self.future_window_num = config['l_seq']  
        self.lstm_batch_size = config['lstm_batch_size']
        self.num_layers = config['num_layers']
        self.device = config['device']
        self.look_back_num = config['look_back_num']
        self.lstm_dropout = config['lstm_dropout']

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.num_hidden_units_lstm, num_layers=self.num_layers, batch_first=True, dropout=self.lstm_dropout) 
        self.fc = nn.Linear(self.num_hidden_units_lstm, self.input_size)
        
    def forward(self, x, targets):  # use Teacher forcing
        x = x.view(x.size(0), -1, self.input_size)
        
        h_0 = torch.zeros(self.num_layers, x.size(0), self.num_hidden_units_lstm).to(self.device)  # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.num_hidden_units_lstm).to(self.device)  # cell state
        
        # get the context vector from LSTM encoder
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        # initialize LSTM decoder input with the last timestamp's value
        lstm_decoder_input = x[:, -1, :].unsqueeze(1)
        
        # collect LSTM decoder's outputs
        lstm_decoder_outputs = []
        
        for t in range(self.l_win): # loop for one window_length (train for prediction ability)
            lstm_output, (h_n, c_n) = self.lstm(lstm_decoder_input, (h_n, c_n))
            next_step = self.fc(lstm_output)
            
            lstm_decoder_outputs.append(next_step)
            
            # Teacher Forcing
            lstm_decoder_input = targets[:, t, :].unsqueeze(1)
            
        # concat lstm decoder outputs
        lstm_decoder_outputs = torch.cat(lstm_decoder_outputs, dim=1)
        
        return lstm_decoder_outputs

    def lstm_infer(self, x): # NO teacher forcing
        x = x.view(x.size(0), -1, self.input_size)
        
        h_0 = torch.zeros(self.num_layers, x.size(0), self.num_hidden_units_lstm).to(self.device)  # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.num_hidden_units_lstm).to(self.device)  # cell state

        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        lstm_decoder_input = x[:, -1, :].unsqueeze(1)
        
        lstm_decoder_outputs = []
        
        for t in range(self.l_win): # loop for one window_length (infer for prediction)
            lstm_output, (h_n, c_n) = self.lstm(lstm_decoder_input, (h_n, c_n))
            next_step = self.fc(lstm_output)
            
            lstm_decoder_outputs.append(next_step)
            
            # NO teachr forcing
            lstm_decoder_input = next_step
            
        # concat lstm decoder outputs
        lstm_decoder_outputs = torch.cat(lstm_decoder_outputs, dim=1)
        
        return lstm_decoder_outputs
    