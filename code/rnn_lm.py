import wiki
import pickle
import random 

import torch
import torch.nn as nn
from torch.autograd import Variable

import time
import math
from collections import Counter
from tqdm import tqdm

DATA_DIR       = '../data/wiki/rnn_lm/'
TOKENIZER_FILE = 'tokenizer.pickle'
TRAIN_FILE     = 'train.pickle'
VAL_FILE       = 'val.pickle'
TEST_FILE      = 'test.pickle'


class DataSet:

    def __init__(self, train, val, test, tokenizer, batch_size, max_seq_length):
        self.tokenizer = tokenizer 
        print("Encoding data splits...")
        self.train = train 
        self.val   = val 
        self.test  = test
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

    @classmethod
    def load(cls, batch_size, max_seq_length):
        with open(DATA_DIR + TRAIN_FILE, 'rb') as f:
            train = pickle.load(f)
        with open(DATA_DIR + VAL_FILE, 'rb') as f:
            val = pickle.load(f)
        with open(DATA_DIR + TEST_FILE, 'rb') as f:
            test = pickle.load(f)
        with open(DATA_DIR + TOKENIZER_FILE, 'rb') as f:
            token_list = pickle.load(f)
            tokenizer = Tokenizer(token_list)
        return cls(train, val, test, tokenizer, batch_size, max_seq_length)

    def save(self):
        with open(DATA_DIR + TRAIN_FILE, 'wb') as f:
            pickle.dump(self.train, f)
        with open(DATA_DIR + VAL_FILE, 'wb') as f:
            pickle.dump(self.val, f)
        with open(DATA_DIR + TEST_FILE, 'wb') as f:
            pickle.dump(self.test, f)
        with open(DATA_DIR + TOKENIZER_FILE, 'wb') as f:
            pickle.dump(self.tokenizer._tokens, f)

    @classmethod
    def from_corpus(cls, max_tokens, batch_size, max_seq_length, sample=None):
        """
        use `sample` to limit the number of posts for testing purposes.
        """
        corpus = wiki.WikiCorpus.from_corpus_files(sample)
         # flatten tokens list since we don't need sentence tokenization for training
        posts = [[token for sentence in post.tokens for token in sentence] for post in corpus.posts.values()]
        random.shuffle(posts)
        num_posts = len(posts)
        print('{} total posts'.format(num_posts))
        print('-' * 38 + 'sample posts' + '-' * 39)
        for post in posts[0:10]:
            print(' '.join(post[:20]) + '...')
        print('-' * 89)

        def split_data(start_percent, end_percent):
            start_split = (num_posts // 100) * start_percent
            end_split   = (num_posts // 100) * end_percent
            return posts[start_split:end_split]

        train = split_data(0, 80)
        val = split_data(80, 90)
        test = split_data(90, 100)

        tokenizer = Tokenizer.fit(train, max_tokens)

        train = tokenizer.encode_texts(train)
        val   = tokenizer.encode_texts(val)
        test  = tokenizer.encode_texts(test)

        return cls(train, val, test, tokenizer, batch_size, max_seq_length)

    def get_split(self, split):
        if split == 'train':
            return self.train
        elif split == 'val':
            return self.val
        elif split == 'test':
            return self.test
        else:
            raise ValueError("No split called {}".format(split))

    def count_batches(self, split):
        return len(self.get_split(split)) // self.batch_size // self.max_seq_length

    def batches(self, split):

        data = self.get_split(split)
        evaluation = split in ('val', 'test')

        # Work out how cleanly we can divide the dataset into batches
        nbatch = data.size(0) // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * self.batch_size)
        # Evenly divide the data across the batches.
        data = data.view(self.batch_size, -1).t().contiguous()

        if cuda:
            data = data.cuda()

        for i in range(0, data.size(0) - 1, max_seq_length):
            seq_len = min(self.max_seq_length, len(data) - 1 - i)
            inputs = Variable(data[i:i+seq_len], volatile=evaluation)
            targets = Variable(data[i+1:i+1+seq_len].view(-1))
            yield inputs, targets 


class Tokenizer:                                                                                           
    END_OF_TEXT = '<END OF TEXT>'
    UNKNOWN_TOKEN = '<UNKNOWN TOKEN>'
        
    def __init__(self, tokens):                                                                          
        self._tokens = tokens                                                                            
        self._index_map = {t: i for i, t in enumerate(self._tokens)}
        self.num_tokens = len(self._tokens)
        self.UNKNOWN_TOKEN_INDEX = self._index_map[Tokenizer.UNKNOWN_TOKEN]
        self.END_OF_TEXT_INDEX = self._index_map[Tokenizer.END_OF_TEXT]
                                                                                                         
    def token_to_index(self, t):                                                                               
        return self._index_map.get(t, self.UNKNOWN_TOKEN_INDEX)
                                                                                                         
    def index_to_token(self, i):                                                                               
        return self._tokens[i]
    
    @classmethod                                                                                         
    def fit(cls, texts, max_tokens):        
        print("Fitting tokenizer to texts...")        
        token_count = Counter()
        for text in tqdm(texts):
            for token in text:                                                                           
                token_count[token] += 1                
        aux_tokens = [Tokenizer.END_OF_TEXT, Tokenizer.UNKNOWN_TOKEN]
        most_common = token_count.most_common(max_tokens - len(aux_tokens))
        # TODO: add some stats about how many tokens were excluded
        print("{} tokens indexed of {} total tokens".format(len(most_common), len(token_count)))
        print("'{}' is the most common indexed word ({} count).".format(*most_common[0]))
        print("'{}' is the least common indexed word ({} count).".format(*most_common[-1]))
        token_list = aux_tokens + [item[0] for item in most_common]
        return cls(token_list)

    def encode_texts(self, texts):        
        num_tokens = sum([len(text) + 1 for text in texts])
        data = torch.LongTensor(num_tokens)        
        i = 0
        for text in tqdm(texts):
            for token in text:
                data[i] = self.token_to_index(token)
                i += 1
            data[i] = self.token_to_index(Tokenizer.END_OF_TEXT)
            i += 1
        return data

    
class LSTM(nn.Module):
    """ https://github.com/pytorch/examples/tree/master/word_language_model """

    def __init__(self, num_tokens, embedding_dim, hidden_dim, num_layers, dropout):
    
        super(LSTM, self).__init__()
     
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Layers
        self.dropout = nn.Dropout(dropout) # dropout for first & final layer 
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
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
        return (Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()),
                Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()))


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(data, split):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(data.batch_size)
    for inputs, targets in data.batches(split):
        output, hidden = model(inputs, hidden)
        output_flat = output.view(-1, data.tokenizer.num_tokens)
        total_loss += len(inputs) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data.get_split(split))


def train(model, dataset, epochs, learning_rate, clip):

    total_batches = dataset.count_batches('train')

    def train_epoch():
        model.train()
        total_loss = 0
        start_time = time.time()
        hidden = model.init_hidden(dataset.batch_size)
        for batch, (inputs, targets) in tqdm(enumerate(dataset.batches('train')), total=total_batches):
            # detach hidden state 
            hidden = repackage_hidden(hidden)
            model.zero_grad()
            output, hidden = model(inputs, hidden)
            loss = criterion(output.view(-1, dataset.tokenizer.num_tokens), targets)
            loss.backward()
            
            # clip_grad_norm helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            for p in model.parameters():
                p.data.add_(-learning_rate, p.grad.data)
                
            total_loss += loss.data
            
            if batch % log_interval == 0 and batch > 0:                                                  
                cur_loss = total_loss[0] / log_interval                                                  
                elapsed = time.time() - start_time
                tqdm.write('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                           'loss {:5.2f} | ppl {:8.2f}'.format(
                            epoch, batch, total_batches, learning_rate,
                            elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()        

    print('Training model...')
    best_val_loss = None
    learning_rate = initial_learning_rate
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train_epoch()
        val_loss = evaluate(dataset, 'val')
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} '
              '| valid ppl {:8.2f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(DATA_DIR + save_prefix + '_lstm_model.pt', 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            learning_rate /= 4.0

if __name__ == '__main__':

    max_tokens = 10000
    embedding_dim = 200
    hidden_dim = 200
    num_layers = 2
    dropout = 0.5
    initial_learning_rate = 20 
    batch_size = 128
    max_seq_length = 32
    clip = 0.25
    log_interval = 200
    criterion = nn.CrossEntropyLoss()
    cuda = True
    epochs = 10 # 40
    save_prefix = 'nov-15a'

    try:
        dataset = DataSet.load(batch_size, max_seq_length)
    except FileNotFoundError:
        dataset = DataSet.from_corpus(max_tokens, batch_size, max_seq_length)
        dataset.save()
        
    model = LSTM(dataset.tokenizer.num_tokens, embedding_dim, hidden_dim, num_layers, dropout)
    if cuda:
        model.cuda()

    train(model, dataset, epochs, initial_learning_rate, clip)
