import os
import torch
import re

def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

class ImdbLoader(object):
    @staticmethod
    def load(path, neg=50, pos=50):
        positives = []
        negatives = []
        directory = os.path.join(path, 'train', 'pos')
        for idx, filename in enumerate(os.listdir(directory)):
            if idx >= pos:
                break
            positives.append(open(os.path.join(directory, filename), 'r').read())

        directory = os.path.join(path, 'train', 'neg')
        for idx, filename in enumerate(os.listdir(directory)):
            if idx >= neg:
                break
            negatives.append(open(os.path.join(directory, filename), 'r').read())

        positives = [striphtml(positive) for positive in positives]
        negatives = [striphtml(negative) for negative in negatives]
        return positives, negatives

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

if __name__ == "__main__":
    import pdb
    positives, negatives = ImdbLoader.load('data/aclImdb', 20, 40)
    pdb.set_trace()
