import torch
import torch.nn as nn
from .layers.Embed import DataEmbedding


class Transformer(nn.Module):

    def __init__(self, in_size, pred_len, out_size, dropout=0.1, d_model=256, n_heads=2, n_layers=2, d_ff=128, activation='gelu'):
        super(Transformer, self).__init__()
        self.in_size = in_size
        self.pred_len = pred_len
        self.out_size = out_size
        self.dropout = dropout
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.activation = activation
        # Embedding
        self.enc_embedding = DataEmbedding(self.in_size, self.d_model, self.dropout)
        # Encoder
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_model*4, dropout=self.dropout, activation=self.activation, batch_first=True), num_layers=self.n_layers)
        # Projection
        self.linear1 = nn.Linear(self.d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, self.out_size)

    def forward(self, x_enc):
        enout_size = self.enc_embedding(x_enc)
        enout_size = self.encoder(enout_size)
        enout_size = torch.tanh(self.linear1(enout_size))
        enout_size = self.linear2(enout_size)
        return enout_size[:, -self.pred_len:, :]  #[B, L, D]
