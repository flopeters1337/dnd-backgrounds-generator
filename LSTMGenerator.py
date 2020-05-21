import torch
import torch.nn as nn
from torch.nn.functional import one_hot


class LSTMGenerator(nn.Module):
    def __init__(self, voc_dim, lstm_dim, embedding_dim, max_len, gpu=False):
        """
        Constructor
        :param voc_dim: (int) vocabulary set's size
        :param lstm_dim: (int) size of hidden states for the lstm layer
        :param embedding_dim: (int) dimension of embedding vector
        :param max_len: (int) length of generated sequences
        :param gpu: (bool) specifies if we are using GPU to compute the neural network
        """
        # Call the superclass' constructor
        super(LSTMGenerator, self).__init__()

        self.gpu = gpu

        self.voc_dim = voc_dim
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
        self.softmax = nn.LogSoftmax(dim=-1)  # -1 to infer the correct dimension

        self.init_params()

    def forward(self, input, hidden, require_hidden=False):
        """
        Forward pass of the network
        :param input: (tensor of size batch_size * max_len) Input sequences
        :param hidden: (tensor of size embedding_dim * lstm_dim) Current hidden states for the LSTM layer
        :param require_hidden: (bool) specifies if the method should return the hidden states or not
        :return: either a single tensor of size batch_size * max_len, 'prediction' which is the result of a forward pass
            of the input through the network; or the aforementioned tensor alongside a tensor of size
            embedding_dim * lstm_dim which correspond to the new hidden states of the LSTM layer after the forward pass
        """
        # Transform the input sequences into sequences of latent vectors
        embedding = self.embedding(input)

        if len(input.size()) == 1:  # Particular case when the sequences contain only one element
            embedding = embedding.unsqueeze(1)

        output, hidden = self.lstm(embedding, hidden)
        output = output.contiguous().view(-1, self.lstm_dim)  # -1 to infer the correct dimension
        output = self.dense(output)
        prediction = self.softmax(output)

        if require_hidden:  # Used when sampling the network
            return prediction, hidden
        else:
            return prediction

    def sample(self, n_samples, batch_size, word_0=-1, gen_type='multinom'):
        """
        Sample the network to generate sequences
        :param n_samples: (int) number of sequences to generate
        :param batch_size: (int) size of batches for generation
        :param word_0: (int) index of the word to bootstrap sequence generation. If it is <0, then select a random word
            from the vocabulary.
        :param gen_type: (string) type of generation method to use (either 'multinom' for multinomial sampling or
            'argmax' for maximum likelihood sampling)
        :return: a tensor of size n_samples * max_len which are sequences of indices sampled from the network
        """
        # Compute the number of batches
        if n_samples != batch_size:
            n_batches = n_samples // batch_size + 1
        else:
            n_batches = 1

        samples = torch.zeros(n_batches * batch_size, self.max_len).long()

        # Produce samples by batches
        for batch in range(n_batches):
            hidden = self.init_hidden(batch_size)
            if word_0 < 0:
                # Initialize every sequence with a random word from the vocabulary
                input = torch.randint(low=0, high=self.voc_dim, size=(batch_size,))
            else:
                # Initialize every sequence with 'word_0' as starting token
                input = torch.LongTensor([word_0] * batch_size)
            if self.gpu:
                input = input.cuda()

            # Iterate the generator until we reach the maximum length allowed for the sequence
            for i in range(self.max_len):
                # Forward pass where we keep track of the hidden states of the network
                output, hidden = self.forward(input, hidden, require_hidden=True)

                if gen_type == 'multinom':
                    # Generate the next token in the sequence randomly using the output as a multinomial distribution
                    next_token = torch.multinomial(torch.exp(output), 1)
                elif gen_type == 'argmax':
                    # Choose the most probable token in the sequence deterministically
                    next_token = torch.argmax(torch.exp(output), 1)

                # Append generated ith tokens to batch #'batch'
                samples[batch * batch_size:(batch + 1) * batch_size, i] = next_token.view(-1)

                # Add generated tokens to the input
                input = next_token.view(-1)

        # We need this because the number of samples might not be divisible by the size of batches
        samples = samples[:n_samples]

        return samples

    def PGLoss(self, input, target, reward):
        """
        Compute a pseudo-gradient loss
        :param input: (tensor of size batch_size * max_len) a batch of input sequences
        :param target: (tensor of size batch_size * max_len) a batch of target sequences that the network should learn
            to reproduce
        :param reward: (tensor of size batch_size * 1) list of rewards associated to each input sequence
        :return: loss object to perform an optimization step
        """
        # NOTE: PG = Pseudo-Gradient
        batch_size, max_len = input.size()
        hidden = self.init_hidden(batch_size)

        output = self.forward(input, hidden).view(batch_size, self.max_len, self.voc_dim)
        onehot_target = one_hot(target, self.voc_dim).float()
        prediction = torch.sum(output * onehot_target, dim=-1)
        loss = -torch.sum(prediction * reward)

        return loss

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad:
                # Initialize all parameters using a normal distribution N(0;1)
                torch.nn.init.normal_(param, mean=0, std=1)

    def init_hidden(self, batch_size):
        # Initialize all hidden states to 0
        h = torch.zeros(1, batch_size, self.lstm_dim)
        c = torch.zeros(1, batch_size, self.lstm_dim)

        if self.gpu:
            return h.cuda(), c.cuda()
        else:
            return h, c
