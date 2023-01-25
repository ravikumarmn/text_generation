import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self,n_vocab,args):
        super(Model, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab+1,
            embedding_dim=args["EMB_DIM"],
        )
        self.lstm = nn.LSTM(
            input_size=self.args["EMB_DIM"],
            hidden_size=self.args["N_HIDDEN"],
            num_layers=self.args["NUM_LAYERS"],
            batch_first = True
        )
        self.fc = nn.Linear(args["N_HIDDEN"], n_vocab+1)

    def forward(self, x,y=None):
        embed = self.embedding(x) 
        output,_ = self.lstm(embed)
        logits = self.fc(output)
        return logits
