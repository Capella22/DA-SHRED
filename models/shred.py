# models/shred.py
import torch
import torch.nn as nn

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim,
                          num_layers=n_layers,
                          batch_first=True,
                          bidirectional=bidirectional)
        self.hidden_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        # x: (batch, seq_len, p)
        out, h_n = self.gru(x)
        h_last = h_n[-1]
        return h_last

class ShallowDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_mul=4, out_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * hidden_mul),
            nn.ReLU(),
            nn.LayerNorm(latent_dim * hidden_mul),
            nn.Linear(latent_dim * hidden_mul, out_dim)
        )

    def forward(self, z):
        return self.net(z)

class SHRED(nn.Module):
    """
    GRU-based SHRED model (inspired by the SINDy-SHRED repository).
    - sensor history (time window) -> GRU encoder -> latent z -> shallow decoder -> full-field
    """
    def __init__(self, sensor_dim, latent_dim=64, enc_hidden=128, dec_out_dim=1024, n_layers=1, decoder_hidden_mul=4):
        super().__init__()
        self.encoder = GRUEncoder(sensor_dim, enc_hidden, n_layers=n_layers, bidirectional=False)
        self.h2z = nn.Sequential(
            nn.Linear(self.encoder.hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )
        self.decoder = ShallowDecoder(latent_dim, hidden_mul=decoder_hidden_mul, out_dim=dec_out_dim)

    def encode(self, sensor_hist):
        h = self.encoder(sensor_hist)
        z = self.h2z(h)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, sensor_hist):
        z = self.encode(sensor_hist)
        x_rec = self.decode(z)
        return x_rec, z
