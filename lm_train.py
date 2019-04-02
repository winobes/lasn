# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import logging
import dill
import torchtext as tt

import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./lm_data/wikitalk_all',
                    help='location of the data corpus')
parser.add_argument('--vocab', type=str, default='./lm_data/wikitalk_all/vocab.dill',
                    help='location of the fitted vocab (torchtext Field object)')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=int, default=None,
                    help='choose CUDA device')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--stop-at-ppl', type=float, default=None,
                    help='Stop training once validation perplexity reaches this value.')
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    handlers = [
        logging.FileHandler(os.path.join(args.data, 'training.log')),
        logging.StreamHandler()
    ])

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if args.cuda is None:
        logging.warning("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:{}".format(args.cuda) if args.cuda is not None else "cpu")

###############################################################################
# Load data
###############################################################################

text_field = torch.load(os.path.join(args.vocab), pickle_module=dill)
train_data, val_data, test_data = tt.datasets.LanguageModelingDataset.splits(path=args.data,
        train='train.txt', validation='valid.txt', test='test.txt',
        text_field=text_field)
ntokens = len(text_field.vocab)
logging.info("Vocab size: {}".format(ntokens))

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

###############################################################################
# Build the model
###############################################################################

model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(batch_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(args.batch_size)
    with torch.no_grad():
        for batch in batch_iter:
            output, hidden = model(batch.text, hidden)
            output_flat = output.view(-1, ntokens)
            # total_loss += len(batch.text) * criterion(output_flat, batch.target.view(-1)).item()
            total_loss += criterion(output_flat, batch.target.view(-1)).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(batch_iter)


def train(batch_iter):
    # Turn on training mode which enables dropout.
    model.train()
    # model.eval()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    for i, batch in enumerate(batch_iter):
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(batch.text, hidden)
        loss = criterion(output.view(-1, ntokens), batch.target.view(-1))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i , len(batch_iter), lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len):
    logging.info('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_iter = tt.data.BPTTIterator(train_data, batch_size=args.batch_size, bptt_len=args.bptt, device=device)
        train(train_iter)
        val_iter = tt.data.BPTTIterator(val_data, batch_size=args.batch_size, bptt_len = args.bptt, device=device)
        val_loss = evaluate(val_iter)
        ppl = math.exp(val_loss)
        logging.info('-' * 89)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, ppl))
        logging.info('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            if args.stop_at_ppl and ppl <= args.stop_at_ppl:
                break 
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    logging.info('-' * 89)
    logging.info('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_iter = tt.data.BPTTIterator(test_data, batch_size=args.batch_size, bptt_len = args.bptt, device=device)
test_loss = evaluate(test_iter)
logging.info('=' * 89)
logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logging.info('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
