import torch.nn as nn
from collections import Counter 
import wiki

class LSTM(nn.Module):

    """ https://github.com/pytorch/examples/tree/master/word_language_model """

    def __init__(self, num_tokens, embedding_dim, hidden_dim, num_layers, dropout):

        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Layers
        self.dropout = nn.Dropout(dropout) # dropout for first & final layer 
        self.embedding = nn.Embeding(num_tokens, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        self.final = nn.Linear(hidden_dim, num_tokens)

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.final.bias.data.fill_(0)
        self.final.weights.data.uniform_(-initrange, initrange)

    def forward(self, input_text, hidden):
        emb = self.dropout(self.embedding(input_text))
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout(output)
        decoded = self.final(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
       
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero()),
                Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero()))


class Lexicon:

    def __init__(self, tokens):
        self._tokens = tokens
        self._index_map = {t: i for i, t in enumerate(self._tokens)}

    @classmethod
    def fit(cls, texts, max_tokens):
            for token in text:
                token_count[token] += 1
        return cls(max_tokens, token_count.most_common(max_tokens))

    def token_to_index(t):
        return self._index_map[t]

    def index_to_token(i):
        return self._tokens[i]

    def encode_text(self, text):
        tensor = torch.zeros(len(text)).long()
        for i, t in enumerate(text):
            tensor[i] = self.token_to_index(t)
        if use_cuda:
            var = Variable(tensor.cuda())
        else:
            var = Varibale(tensor)
        return var

    def __len__(self):
        return len(self._tokens)


if __name__ == '__main__':

    max_tokens = 10000
    embedding_dim = 200 
    hidden_dim = 200 
    num_layers = 2 
    dropout = 0.5
    learning_rate = 0.005
    batch_size = 128

    corpus = wiki.WikiCorpus.from_corpus_files()
    posts = (post.clean_text for post in corpus.posts.values())
    lexicon = Lexicon.fit(posts, max_tokens)


     


