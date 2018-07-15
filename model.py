import numpy as np
import torch
import torch.nn as nn
import pdb

class LanguageModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.2, tie_weights=False):
        super(LanguageModel, self).__init__()
        self.rnn_model = RNNModel(ntoken, ninp, nhid, nlayers, dropout=0.2)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.rnn_model.encoder.weight

        self.init_weights()

    def init_hidden(self, bsz):
        return self.rnn_model.init_hidden(bsz)

    def forward(self, input, hiddens):
        output, hidden = self.rnn_model(input, hiddens)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

class RNNModel(nn.Module):
    """Container module with an encoder, and a recurrent module"""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.2):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnns = nn.ModuleList([nn.GRU(ninp, nhid, 1, dropout=dropout)])
        self.rnns.extend([nn.GRU(nhid, nhid, 1, dropout=dropout) for i in range(1, nlayers)])

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hiddens):
        output = self.drop(self.encoder(input))
        new_hidden = hiddens.clone()

        for i, rnn in enumerate(self.rnns):
            output, hidden = rnn(output, hiddens[i].unsqueeze(0))
            new_hidden[i] = hidden

        output = self.drop(output)
        return output, new_hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def flatten_parameters(self):
        [rnn.flatten_parameters() for rnn in self.rnns]

def main():
    batch_size = 64
    sequence_length = 50
    inp = torch.autograd.Variable(torch.randint(400, (sequence_length, batch_size)).long())
    model = RNNModel(400, 300, 300, 3)
    hidden = model.init_hidden(batch_size)
    output = model(inp, hidden)

if __name__ == "__main__":
    main()
