import wiki
import pickle
import random 
import os
import argparse
import json

import torch
import torch.nn as nn
from torch.autograd import Variable

import time
import math
from collections import Counter
from tqdm import tqdm

criterion = nn.CrossEntropyLoss()  # log of perplexity
log_interval = 200
cuda = True

class DataSet:

    def __init__(self, train, val, test, tokenizer, batch_size, max_seq_len):
        self.tokenizer = tokenizer 
        self.train = train 
        self.val   = val 
        self.test  = test
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    @classmethod
    def load(cls, data_dir, batch_size, max_seq_len):
        with open(data_dir + 'train.pickle', 'rb') as f:
            train = pickle.load(f)
        with open(data_dir + 'val.pickle', 'rb') as f:
            val = pickle.load(f)
        with open(data_dir + 'test.pickle', 'rb') as f:
            test = pickle.load(f)
        tokenizer = Tokenizer.load(data_dir)
        return cls(train, val, test, tokenizer, batch_size, max_seq_len)

    def save(self, data_dir):
        with open(data_dir + 'train.pickle', 'wb') as f:
            pickle.dump(self.train, f)
        with open(data_dir + 'val.pickle', 'wb') as f:
            pickle.dump(self.val, f)
        with open(data_dir + 'test.pickle', 'wb') as f:
            pickle.dump(self.test, f)
        with open(data_dir + 'tokenizer.pickle', 'wb') as f:
            pickle.dump(self.tokenizer._tokens, f)

    @classmethod
    def from_corpus(cls, max_tokens, batch_size, max_seq_len, data_dir, sample=None):
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

        print("Encoding data splits...")
        train = tokenizer.encode_texts(train)
        val   = tokenizer.encode_texts(val)
        test  = tokenizer.encode_texts(test)

        return cls(train, val, test, tokenizer, batch_size, max_seq_len)

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
        return len(self.get_split(split)) // self.batch_size // self.max_seq_len

    def batches(self, split):

        data = self.get_split(split)
        data = batchify(data, self.batch_size)
        evaluation = split in ('val', 'test')

        if cuda:
            data = data.cuda()

        return generate_sequences(data, self.max_seq_len, evaluation)


def generate_sequences(data, max_seq_len, evaluation):

    for i in range(0, data.size(0) - 1, max_seq_len):
        seq_len = min(max_seq_len, len(data) - 1 - i)
        inputs = Variable(data[i:i+seq_len], volatile=evaluation)
        targets = Variable(data[i+1:i+1+seq_len].view(-1))
        yield inputs, targets 


def batchify(data, batch_size, cuda=True):
    # Work out how cleanly we can divide the dataset into batch_size parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data

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
        print("{} tokens indexed of {} total tokens".format(len(most_common), len(token_count)))
        print("'{}' is the most common indexed word ({} count).".format(*most_common[0]))
        print("'{}' is the least common indexed word ({} count).".format(*most_common[-1]))
        token_list = aux_tokens + [item[0] for item in most_common]
        for token in token_list[:10]:
            print(token)
        return cls(token_list)

    @classmethod
    def load(cls, data_dir):
        with open(data_dir + 'tokenizer.pickle', 'rb') as f:
            tokens = pickle.load(f)
        return cls(tokens)

    def encode_texts(self, texts):        
        num_tokens = sum([len(text) + 1 for text in texts])
        data = torch.LongTensor(num_tokens)        
        i = 0
        for text in texts:
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

    @classmethod
    def load(cls, tokenizer, model_dir):
        with open(model_dir + 'train_params.txt') as f:
            args = json.load(f)
        print(args)
        model = cls(
                tokenizer.num_tokens, 
                args['embedding_dim'], 
                args['hidden_dim'], 
                args['num_layers'], 
                args['dropout'])
        model.load_state_dict(torch.load(model_dir + 'model.pt'))
        return model

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir + 'model.pt')

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


def perplexity(model, tokenizer, tokens):
    batch_size = len(tokens)
    data = tokenizer.encode_texts([tokens])
    data = data.view(len(data), -1).t().contiguous()

    model.eval()
    hidden = model.init_hidden(batch_size)


    inputs = Variable(data.narrow(1,0,batch_size))
    targets = Variable(data.narrow(1,1,batch_size).view(-1))



    output, hidden = model(inputs, hidden)
    output_flat = output.view(-1, tokenizer.num_tokens)
    loss = criterion(output_flat, targets).data
    hidden = repackage_hidden(hidden)
    return math.exp(loss[0])


def evaluate(model, data, num_tokens):

    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(32)
    data = batchify(data, 32)
    if cuda:
        data = data.cuda()

    for inputs, targets in generate_sequences(data, 1024, False): 
        output, hidden = model(inputs, hidden)
        output_flat = output.view(-1, num_tokens)
        total_loss += len(inputs) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)

    return total_loss[0] / len(data)


def train(model, dataset, epochs, initial_learning_rate, clip, job_dir):

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
        val_loss = evaluate(model, dataset.get_split('val'), dataset.tokenizer.num_tokens)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} '
              '| valid ppl {:8.2f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            model.save(job_dir)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            learning_rate /= 4.0


class Evaluator:
    """ wraps `perplexity` function to re-use model & tokenizer for many utterances """
    def __init__(self, model_dir):

        if not model_dir[-1] == '/':
            model_dir += '/'
        
        self.tokenizer = Tokenizer.load(model_dir)
        self.model = LSTM.load(self.tokenizer, model_dir)

    def perplexity(self, tokenized_text):
        return perplexity(self.model, self.tokenizer, tokenized_text)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Create a LSTM from the wiki comment corpus')

    parser.add_argument('job', type=str)
    parser.add_argument('job_dir', type=str)

    parser.add_argument('--embedding-dim', type=int, default=200)
    parser.add_argument('--hidden-dim', type=int, default=200)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--initial-learning-rate', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-seq-len', type=int, default=32)
    parser.add_argument('--max-tokens', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--sample', type=int, default=None)

    args = parser.parse_args()

    job_dir        = '../data/wiki/rnn_lm/' + args.job_dir
    if not job_dir[-1] == '/':
        job_dir += '/'

    model_file     = job_dir + 'model.pt'
    params_file    = job_dir + 'train_params.txt'

    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    try:
        dataset = DataSet.load(job_dir, args.batch_size, args.max_seq_len)
    except FileNotFoundError:
       dataset = DataSet.from_corpus(
                args.max_tokens, 
                args.batch_size, 
                args.max_seq_len, 
                job_dir,
                sample=args.sample)
       dataset.save(job_dir)


    if args.job == 'train':

        print(args.__dict__)
        with open(params_file, 'w') as f:
            json.dump(args.__dict__, f)

        model = LSTM(
                dataset.tokenizer.num_tokens, 
                args.embedding_dim, 
                args.hidden_dim, 
                args.num_layers, 
                args.dropout)
        if cuda:
            model.cuda()

        train(model, dataset, args.epochs, args.initial_learning_rate, args.clip, job_dir)


    elif args.job == 'eval':

        model = LSTM.load(job_dir)

        evaluate(model, dataset.get_split('test'), dataset.tokenizer.num_tokens)
