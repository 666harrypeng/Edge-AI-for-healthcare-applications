import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEmodel(nn.Module):
    def __init__(self, config):
        super(VAEmodel, self).__init__()
        self.config = config
        self.l_win = config['l_win']    
        self.n_channel = config['n_channel']    
        self.code_size = config['code_size']   
        self.num_hidden_units = config['num_hidden_units']

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_channel, self.num_hidden_units // 16, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(self.num_hidden_units // 16, self.num_hidden_units // 8, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(self.num_hidden_units // 8, self.num_hidden_units // 4, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(self.num_hidden_units // 4, self.num_hidden_units // 2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(self.num_hidden_units // 2, self.num_hidden_units, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU()
        )

        self.conv_mean = nn.Conv2d(self.num_hidden_units, self.code_size, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.conv_logvar = nn.Conv2d(self.num_hidden_units, self.code_size, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.code_size, self.num_hidden_units // 4, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hidden_units // 4, self.num_hidden_units // 8, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hidden_units // 8, self.num_hidden_units // 16, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hidden_units // 16, self.num_hidden_units // 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hidden_units // 32, 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        )

    def encode(self, x):
        assert x.shape[1:] == (self.config['n_channel'], self.config['l_win'], 1), "VAE Encoder input shape is not correct!"
        x = self.encoder(x)
        mean = self.conv_mean(x)
        logvar = self.conv_logvar(x)
        return mean, logvar

    def decode(self, z):
        assert z.shape[1:] == (self.config['code_size'], self.config['l_win'], 1), "VAE Decoder input shape is not correct!"
        x_reconstructed = self.decoder(z)
        return x_reconstructed
    
    def forward(self, x):
        assert x.shape[1:] == (self.config['n_channel'], self.config['l_win'], 1), "VAE input shape is not correct!"
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
        self.seq_length = config['l_win']   
        self.num_hidden_units_lstm = config['num_hidden_units_lstm']
        self.input_size = config['code_size']   
        self.future_steps = config['l_seq']
        self.batch_size = config['batch_size']
        self.num_layers = config['num_layers'] 

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.num_hidden_units_lstm, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.num_hidden_units_lstm, self.input_size)  

    def forward(self, x):
        assert x.shape[1:] == (self.seq_length - self.future_steps, self.input_size), "LSTM forward input shape is not correct!"
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        predictions = []
        
        # Initialize the input for future steps as the last time step from input
        decoder_input = x[:, -1, :].unsqueeze(1)

        for _ in range(self.future_steps):
            # Pass the last output into the LSTM
            lstm_out, (h_n, c_n) = self.lstm(decoder_input, (h_n, c_n))

            next_step = self.fc(lstm_out[:, -1, :])  
            predictions.append(next_step.unsqueeze(1))  
            decoder_input = next_step.unsqueeze(1)

        predictions = torch.cat(predictions, dim=1)  

        return predictions

