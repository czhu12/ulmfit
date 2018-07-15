import torch
from torch import nn
import pdb
import pickle
from typing import List


class ClassificationModel(nn.Module):
    def __init__(self, rnn, classes):
        super(ClassificationModel, self).__init__()
        self.rnn = rnn
        hidden_size = self.rnn.rnns[-1].hidden_size
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, classes),
            nn.ReLU(),
        )

    def forward(self, x, hidden, lengths):
        outputs, _ = self.rnn(x, hidden)

        pdb.set_trace()
        masks = (lengths-1).view(1, -1, 1).expand(output.size(0), output.size(1), output.size(2))
        output = outputs.gather(0, masks)[0]
        output = self.transform(output)
        return output

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

def classifier_training(model, texts, y, dictionary):
    batch_size = 2
    lengths = torch.tensor([len(text.split()) for text in texts])
    sequences = torch.tensor(_pad_sequences(_serialize(texts, dictionary)))
    sequences = torch.transpose(sequences, 1, 0)
    hidden = model.init_hidden(batch_size)
    model(sequences, hidden, lengths)

def main():
    language_model = torch.load('model.pt', map_location='cpu')
    rnn_model = language_model.rnn_model
    model = ClassificationModel(rnn_model, 2)
    with open('dict.pickle', 'rb') as f:
        dictionary = pickle.load(f)
    #sequence_length = 50
    #inp = torch.autograd.Variable(torch.randint(400, (sequence_length, batch_size)).long())
    #model = RNNModel(400, 300, 300, 3)
    texts = [
        'man i really did not like this movie because it was so boring',
        'this movie sucked, it was painful to watch, it really was',
    ]
    y = torch.tensor([
        [0, 1],
        [1, 0],
    ])
    classifier_training(model, texts, y, dictionary)
main()
