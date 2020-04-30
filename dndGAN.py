import torch
import torch.nn as nn

# Enables GPU computing to speed up network training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTMGenerator(nn.Module):
    def __init__(self, voc_dim, lstm_dim, embedding_dim, max_len, gpu=False):
        """
        Constructor
        :param voc_dim:
        :param lstm_dim:
        :param embedding_dim:
        :param gpu: (bool) number of GPUs to use for computation
        """
        # Call the superclass' constructor
        super(LSTMGenerator, self).__init__()

        self.gpu = gpu

        self.lstm_dim = lstm_dim
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        # Embedding layer which maps from a vocabulary to a latent vector space and vice versa
        self.embedding = nn.Embedding(voc_dim, embedding_dim)

        # LSTM layer with hidden states
        self.lstm = nn.LSTM(embedding_dim, lstm_dim, batch_first=True)

        # Final dense layer (corresponds to matrix multiplication)
        self.dense = nn.Linear(lstm_dim, voc_dim)

        # Final softmax layer
        self.softmax = nn.Softmax(dim=-1)

        self.init_params()

    def forward(self, input, hidden, require_hidden=False):
        """

        :param input:
        :param hidden:
        :param require_hidden:
        :return:
        """
        embedding = self.embedding(input)
        if len(input.size()) == 1:
            embedding = embedding.unsqueeze(1)

        output, hidden = self.lstm(embedding, hidden)
        output = output.contiguous().view(-1, self.lstm_dim)
        output = self.dense(output)
        prediction = self.softmax(output)

        if require_hidden:
            return prediction, hidden
        else:
            return prediction

    def sample(self, n_samples, batch_size, word_0):
        if n_samples != batch_size:
            n_batches = n_samples // batch_size + 1
        else:
            n_batches = 1

        samples = torch.zeros(n_batches * batch_size, self.max_len).long()

        for batch in range(n_batches):
            hidden = self.init_hidden(batch_size)
            input = torch.LongTensor([word_0] * batch_size)
            if self.gpu:
                input = input.cuda()

            for i in range(self.max_len):
                output, hidden = self.forward(input, hidden, require_hidden=True)
                next_token = torch.multinomial(output, 1)
                samples[batch * batch_size:(batch + 1) * batch_size, 1] = next_token.view(-1)
                input = next_token.view(-1)

        samples = samples[:n_samples]

        return samples

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0, std=1)

    def init_hidden(self, batch_size):
        h = torch.zeros(1, batch_size, self.lstm_dim)
        c = torch.zeros(1, batch_size, self.lstm_dim)

        if self.gpu:
            return h.cuda(), c.cuda()
        else:
            return h, c


def discriminator():
    pass
