import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self,n_vocab,args):
        super().__init__()
        self.args = args

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab+1,
            embedding_dim=args['EMB_DIM']
        )
        self.transformer_model = nn.Transformer(
            d_model=args['EMB_DIM'],
            # nhead=args['N_HEADS'],
            # num_decoder_layers=args['ENCODER_LAYERS'],
            
        )
        self.fc = nn.Linear(args['EMB_DIM'], n_vocab+1)
    
    def forward(self,x,y):
        x_embed = self.embedding(x)
        y_embed = self.embedding(y)
        output = self.transformer_model(x_embed,y_embed)
        logits = self.fc(output)
        return logits