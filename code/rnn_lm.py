import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, max_tokens, embedding_dim, hidden_dim, num_layers, dropout):

        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Layers
        self.dropout = nn.Dropout(dropout) # dropout for first & final layer 
        self.embedding = nn.Embeding(max_num_tokens, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        self.final = nn.Linear(hidden_dim, max_tokens)

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
