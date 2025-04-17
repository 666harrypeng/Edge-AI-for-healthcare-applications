import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = x.squeeze(-1) 
        x = x.permute(0, 2, 1)
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.norm(attn_output)
        output = attn_output.permute(0, 2, 1)
        output = output.unsqueeze(-1)
        return output
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.3)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        out = self.dropout(out)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope=0.2)
        return F.leaky_relu(out + identity, negative_slope=0.2)

class VAEmodel(nn.Module):
    def __init__(self, config):
        super(VAEmodel, self).__init__()
        self.config = config
        self.l_win = config['l_win']    
        self.input_size = config['input_size']    
        self.code_size = config['code_size']    
        self.num_hidden_units_vae = config['num_hidden_units_vae']

        # Initial convolution
        self.initial_conv = nn.Conv2d(self.input_size, self.num_hidden_units_vae // 16, kernel_size=(3, 1), padding=(1, 0))
        self.initial_bn = nn.BatchNorm2d(self.num_hidden_units_vae // 16)
        self.initial_ln = nn.LayerNorm([self.num_hidden_units_vae // 16, self.l_win, 1])
        
        # Residual blocks for encoder
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(self.num_hidden_units_vae // 16, self.num_hidden_units_vae // 8),
            ResidualBlock(self.num_hidden_units_vae // 8, self.num_hidden_units_vae // 4),
            ResidualBlock(self.num_hidden_units_vae // 4, self.num_hidden_units_vae // 2),
            ResidualBlock(self.num_hidden_units_vae // 2, self.num_hidden_units_vae)
        ])
        
        # Multi-head attention
        self.attention = MultiHeadSelfAttention(self.num_hidden_units_vae, num_heads=4)
        self.post_attention_ln = nn.LayerNorm([self.num_hidden_units_vae, self.l_win, 1])
        
        # Latent space
        self.conv_mean = nn.Conv2d(self.num_hidden_units_vae, self.code_size, kernel_size=(3, 1), padding=(1, 0))
        self.conv_logvar = nn.Conv2d(self.num_hidden_units_vae, self.code_size, kernel_size=(3, 1), padding=(1, 0))
        self.latent_ln = nn.LayerNorm([self.code_size, self.l_win, 1])

        # Decoder residual blocks
        self.decoder_residual_blocks = nn.ModuleList([
            ResidualBlock(self.code_size, self.num_hidden_units_vae // 4),
            ResidualBlock(self.num_hidden_units_vae // 4, self.num_hidden_units_vae // 8),
            ResidualBlock(self.num_hidden_units_vae // 8, self.num_hidden_units_vae // 16),
            ResidualBlock(self.num_hidden_units_vae // 16, self.input_size)
        ])
        
        # Final output layer
        self.final_conv = nn.Conv2d(self.input_size, self.input_size, kernel_size=(3, 1), padding=(1, 0))
        self.final_ln = nn.LayerNorm([self.input_size, self.l_win, 1])
        
        # Modality weights for reconstruction loss
        self.modality_weights = torch.tensor(config.get('modality_weights', [0.5, 0.2, 0.3]))

    def encode(self, x):
        assert x.shape[1:] == (self.input_size, self.l_win, 1), "VAE Encoder input shape is not correct!"
        
        # Initial convolution with dual normalization
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_ln(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Multi-head attention with residual connection and layer norm
        attention_out = self.attention(x)
        x = x + attention_out  # Residual connection
        x = self.post_attention_ln(x)
        
        # Latent space projections with layer norm
        mean = self.latent_ln(self.conv_mean(x))
        logvar = self.latent_ln(self.conv_logvar(x))
        
        return mean, logvar

    def decode(self, z):
        assert z.shape[1:] == (self.code_size, self.l_win, 1), "VAE Decoder input shape is not correct!"
        
        # Residual blocks for decoder
        x = z
        for block in self.decoder_residual_blocks:
            x = block(x)
            
        # Final output with layer norm
        x = self.final_conv(x)
        x_reconstructed = self.final_ln(x)
        
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

    def weighted_reconstruction_loss(self, x, x_reconstructed):
        # Ensure weights are on the correct device
        weights = self.modality_weights.to(x.device)
        
        # Calculate MSE loss for each modality
        mse_loss = F.mse_loss(x, x_reconstructed, reduction='none')
        
        # Apply weights to each modality and take mean over batch and time dimensions
        weighted_loss = mse_loss * weights.view(1, -1, 1, 1)
        
        return weighted_loss.sum()


class LSTMmodel(nn.Module):
    def __init__(self, config):
        super(LSTMmodel, self).__init__()
        self.l_win = config['l_win']   
        self.num_hidden_units_lstm = config['num_hidden_units_lstm']
        self.input_size = config['code_size']   
        self.l_seq = config['l_seq']  # default 1 (num of future windows to predict)
        self.batch_size = config['batch_size']
        self.num_layers = config['num_layers']  
        self.device = config['device']
        self.look_back_num = config['look_back_num']
        self.lstm_dropout = config['lstm_dropout']
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.num_hidden_units_lstm, num_layers=self.num_layers, batch_first=True, dropout=self.lstm_dropout)
        self.fc = nn.Linear(self.num_hidden_units_lstm, self.input_size)

    def forward(self, x, targets):  # use teacher forcing
        assert x.shape[1:] == (self.look_back_num, self.l_win, self.input_size), "x shape is not correct at the beginning of LSTM"
        assert targets.shape[1:] == (self.l_win, self.input_size), "targets shape is not correct at the beginning of LSTM"
        
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