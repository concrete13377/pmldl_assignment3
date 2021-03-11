import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 12


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        print(f'input.shape: {input.shape}')
        print(f'hidden.shape: {hidden.shape}')
        embedded = self.embedding(input).view(1, 1, -1)
        print(f'embedded.shape: {embedded.shape}')
        output = embedded
        output, hidden = self.gru(output, hidden)
        print(f'output.shape: {output.shape}')
        print(f'hidden.shape: {hidden.shape}')
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class EncoderLSTM(nn.Module):
    def __init__(self, num_words, word_dim, num_chars, char_dim, num_tag, lstm_emb=64):
        super().__init__()
        # Initializing Word Embedder
        self.word_emb = nn.Embedding(num_words, word_dim)
        # Initializing Char Embedder
        self.char_emb = nn.Embedding(num_chars, char_dim)
        # Initializing 1D CNN
        self.char_conv = nn.Conv1d(char_dim, char_dim, kernel_size=3, padding=1)
        # Initializing Activation Function
        self.relu = nn.ReLU()
        # Initializing LSTM
        self.lstm = nn.LSTM(word_dim + char_dim, lstm_emb, bidirectional=True)
        # Initializing DropOut
        self.drop_out = nn.Dropout(0.2)
        # Initializing Linear
        self.linear = nn.Linear(in_features=2 * lstm_emb, out_features=num_tag)

    def forward(self, chars, sent):
        # Make embeddings from words in the sentence
        word_emb = self.word_emb(sent)  # [num_words x word_dim]
        # Make embeddings from chars in the word in the sentence
        char_emb = self.char_emb(chars)  # [num_words x num_chars x char_dim]
        char_emb = char_emb.permute(0, 1, 3, 2)  # [num_words x char_dim x num_chars]
        # 1D CNN
        char_emb = self.char_conv(char_emb[0])  # [num_words x char_dim x num_chars]
        char_emb = self.relu(char_emb)  # [num_words x char_dim x num_chars]
        char_emb = torch.mean(char_emb, 2)  # [num_words x char_dim x 1]
        word_emb = word_emb[0]  # [num_words x char_dim]
        # Words and chars embeddings concatenation
        word_emb = torch.cat([word_emb, char_emb], 1)  # [num_words x (word_dim + char_dim)]
        word_emb = torch.unsqueeze(word_emb, 1)  # [num_words x 1 x (word_dim + char_dim)]
        # LSTM
        word_emb, _ = self.lstm(word_emb)  # [num_words x 1 x (word_dim + char_dim)]
        word_emb = self.drop_out(word_emb)
        word_emb = self.linear(word_emb)  # [num_tag x num_words x 1]

        return word_emb.permute(1, 0, 2)[0]  # [num_words x num_tag]


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=max_len):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = 12

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
