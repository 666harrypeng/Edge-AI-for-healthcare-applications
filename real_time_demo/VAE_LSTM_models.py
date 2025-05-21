import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEmodel(nn.Module):
    def __init__(self, config):
        super(VAEmodel, self).__init__()
        self.config = config
        self.l_win = config['l_win']    
        self.input_size = config['input_size']    
        self.code_size = config['code_size']   
        self.num_hidden_units_vae = config['num_hidden_units_vae']

        # Encoder (CNN) for each time step
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_size, self.num_hidden_units_vae // 16, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(self.num_hidden_units_vae // 16, self.num_hidden_units_vae // 8, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(self.num_hidden_units_vae // 8, self.num_hidden_units_vae // 4, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(self.num_hidden_units_vae // 4, self.num_hidden_units_vae // 2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(self.num_hidden_units_vae // 2, self.num_hidden_units_vae, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU()
        )
        
        # Latent space
        self.conv_mean = nn.Conv2d(self.num_hidden_units_vae, self.code_size, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.conv_logvar = nn.Conv2d(self.num_hidden_units_vae, self.code_size, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        # Decoder (CNN-Transpose)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.code_size, self.num_hidden_units_vae // 4, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hidden_units_vae // 4, self.num_hidden_units_vae // 8, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hidden_units_vae // 8, self.num_hidden_units_vae // 16, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hidden_units_vae // 16, self.num_hidden_units_vae // 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hidden_units_vae // 32, self.input_size, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        )

    def encode(self, x):
        assert x.shape[1:] == (self.input_size, self.l_win, 1), "VAE Encoder input shape is not correct!"
        
        x = self.encoder(x)
        
        mean = self.conv_mean(x)
        
        logvar = self.conv_logvar(x)
        return mean, logvar

    def decode(self, z):
        assert z.shape[1:] == (self.code_size, self.l_win, 1), "VAE Decoder input shape is not correct!"
        x_reconstructed = self.decoder(z)
        return x_reconstructed
    
    def forward(self, x):
        assert x.shape[1:] == (self.input_size, self.l_win, 1), "VAE input shape is not correct!"
        mean, logvar = self.encode(x)
        
        z = self.reparameterize(mean, logvar)
        
        x_reconstructed = self.decode(z)
        return x_reconstructed, mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


class LSTMmodel(nn.Module):
    def __init__(self, config):
        super(LSTMmodel, self).__init__()
        self.l_win = config['l_win']   
        self.num_hidden_units_lstm = config['num_hidden_units_lstm']
        self.input_size = config['code_size']   # Latent space dimension from VAE  
        self.l_seq = config['l_seq']  
        self.batch_size = config['batch_size']
        self.num_layers = config['num_layers']  # LSTM layers (default 3)
        self.device = config['device']
        self.look_back_num = config['look_back_num']
        self.lstm_dropout = config['lstm_dropout']
        '''
        With batch_first=True, PyTorch expects input to be in the shape: (batch_size, ...)
        '''
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.num_hidden_units_lstm, num_layers=self.num_layers, batch_first=True, dropout=self.lstm_dropout)
        self.fc = nn.Linear(self.num_hidden_units_lstm, self.input_size)

    def forward(self, x, targets):  # use teacher forcing
        assert x.shape[1:] == (self.look_back_num, self.l_win, self.input_size), "x shape is not correct at the beginning of LSTM"
        assert targets.shape[1:] == (self.l_win, self.input_size), "targets shape is not correct at the beginning of LSTM"
        
        ''' x reshape
        (batch_size, look_back_num, window_length, code_size) -> (batch_size, look_back_ALL, code_size)
        '''
        
        x = x.view(x.size(0), -1, self.input_size)
        
        h_0 = torch.zeros(self.num_layers, x.size(0), self.num_hidden_units_lstm).to(self.device)  # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.num_hidden_units_lstm).to(self.device)  # cell state
        
        # get the context vector from LSTM Encoder
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        # initialize LSTM decoder input with the last timestamp's value
        lstm_decoder_input = x[:, -1, :].unsqueeze(1)
        # collect LSTM decoder's outputs
        lstm_decoder_outputs = []
        
        for t in range(self.l_win): # loop for 1 window_length
            lstm_output, (h_n, c_n) = self.lstm(lstm_decoder_input, (h_n, c_n))
            next_step = self.fc(lstm_output)
            
            lstm_decoder_outputs.append(next_step)
            
            ### Teacher forcing: use true embedding as next input
            lstm_decoder_input = targets[:, t, :].unsqueeze(1)
            
        # concat lstm decoder outputs
        lstm_decoder_outputs = torch.cat(lstm_decoder_outputs, dim=1)
        
        return lstm_decoder_outputs

    def lstm_infer(self, x):    # use NO teacher forcing
        assert x.shape[1:] == (self.look_back_num, self.l_win, self.input_size), "x shape is not correct at the beginning of LSTM"
        
        ''' x reshape
        (batch_size, look_back_num, window_length, code_size) -> (batch_size, look_back_ALL, code_size)
        '''
        
        x = x.view(x.size(0), -1, self.input_size)
        
        h_0 = torch.zeros(self.num_layers, x.size(0), self.num_hidden_units_lstm).to(self.device)  # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.num_hidden_units_lstm).to(self.device)  # cell state
        
        # get the context vector from LSTM Encoder
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        # initialize LSTM decoder input with the last timestamp's value
        lstm_decoder_input = x[:, -1, :].unsqueeze(1)
        # collect LSTM decoder's outputs
        lstm_decoder_outputs = []
        
        for t in range(self.l_win): # loop for 1 window_length
            lstm_output, (h_n, c_n) = self.lstm(lstm_decoder_input, (h_n, c_n))
            next_step = self.fc(lstm_output)
            
            lstm_decoder_outputs.append(next_step)

            ### No teacher forcing: direct auto-regression, use own output as next input
            lstm_decoder_input = next_step

        # concat lstm decoder outputs
        lstm_decoder_outputs = torch.cat(lstm_decoder_outputs, dim=1)
        
        return lstm_decoder_outputs