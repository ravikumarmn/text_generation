import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self,n_vocab,args):
        super(Model, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=args["EMB_DIM"],
        )
        self.lstm = nn.LSTM(
            input_size=self.args["EMB_DIM"],
            hidden_size=self.args["N_HIDDEN"],
            num_layers=self.args["NUM_LAYERS"],
            batch_first = True
        )
        self.fc = nn.Linear(args["N_HIDDEN"], n_vocab)

    def forward(self, x):
        embed = self.embedding(x)
        # state_h = hidden[0].repeat(1,self.args["BATCH_SIZE"],1)
        # state_c = hidden[1].repeat(1,self.args["BATCH_SIZE"],1)
        output,_ = self.lstm(embed)
        logits = self.fc(output)
        return logits

    def init_state(self):
        return (torch.zeros(self.args["NUM_LAYERS"], 1, self.args["N_HIDDEN"],device='cuda'),
                torch.zeros(self.args["NUM_LAYERS"], 1, self.args["N_HIDDEN"],device="cuda"))