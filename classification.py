import torch
from torch import nn
import numpy as np
import pdb
import pickle
from typing import List
from torch.autograd import Variable
from data import ImdbLoader


class ClassificationModel(nn.Module):
    def __init__(self, rnn, classes):
        super(ClassificationModel, self).__init__()
        self.rnn = rnn
        hidden_size = self.rnn.rnns[-1].hidden_size
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, classes),
            nn.LogSoftmax(),
        )

    def forward(self, x, hidden, lengths):
        outputs, _ = self.rnn(x, hidden)
        masks = (lengths-1).view(1, -1, 1).expand(outputs.size(0), outputs.size(1), outputs.size(2))
        output = outputs.gather(0, masks)[0]
        scores = self.transform(output)
        return scores

    def init_hidden(self, bsz):
        return self.rnn.init_hidden(bsz)

def _pad_sequence(sequence: List[int], target_length: int) -> List[int]:
    assert target_length >= len(sequence), 'target_length is longer than sequence length'
    extra = target_length - len(sequence)
    return sequence + [0] * extra


def _pad_sequences(sequences: List[List[int]]):
    maxlen = max([len(sequence) for sequence in sequences])
    return [_pad_sequence(sequence, maxlen) for sequence in sequences]

def _serialize(texts, dictionary):
    serialized = []
    mapping = dictionary.word2idx
    for text in texts:
        serialized.append([mapping[word] if word in mapping else mapping['<unk>'] for word in text.split()])
    return serialized

def classifier_training(model, texts, y, dictionary, epochs=500, batch_size=32):
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    for e in range(epochs):
        for b in range(batch_size):
            optimizer.zero_grad()
            lengths = torch.tensor([len(text.split()) for text in texts])
            sequences = torch.tensor(_pad_sequences(_serialize(texts, dictionary)))
            sequences = torch.transpose(sequences, 1, 0)
            hidden = model.init_hidden(batch_size)
            scores = model(Variable(sequences), Variable(hidden), Variable(lengths))
            loss = criterion(scores, Variable(y))
            loss.backward()
            print(loss)
            optimizer.step()

def main():
    language_model = torch.load('model.pt', map_location='cpu')
    rnn_model = language_model.rnn_model
    batch_size = 32
    model = ClassificationModel(rnn_model, batch_size)
    with open('dict.pickle', 'rb') as f:
        dictionary = pickle.load(f)
    #sequence_length = 50
    #inp = torch.autograd.Variable(torch.randint(400, (sequence_length, batch_size)).long())
    #model = RNNModel(400, 300, 300, 3)
    positives, negatives = ImdbLoader.load('data/aclImdb', neg=16, pos=16)
    texts = positives + negatives
    y = torch.from_numpy(np.append(np.zeros(len(positives)), np.ones(len(negatives)))).long()
    classifier_training(model, texts, y, dictionary, batch_size=batch_size)
main()
