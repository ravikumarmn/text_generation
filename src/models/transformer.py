# REFERENCE  : https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import torch.nn as nn
import torch
from torch import Tensor
import math
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self,n_vocab,args):
        super().__init__()
        self.args = args
        self.d_model = args['EMB_DIM']
        self.pos_encoder = PositionalEncoding(self.d_model, args["DROPOUT"])

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab+1,
            embedding_dim=self.d_model
        )
        encoder_layers = nn.TransformerEncoderLayer(self.d_model,\
            args['N_HEADS'],args['N_HIDDEN'], args["DROPOUT"],batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, args['N_LAYERS'])

        self.fc = nn.Linear(self.d_model, n_vocab+1)

    def generate_square_subsequent_mask(self,sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self,src):
        _,seq_len = src.size()
        src_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.fc(output)
        return output