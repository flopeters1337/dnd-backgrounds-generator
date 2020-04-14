import torch
import torch.nn as nn

# Enables GPU computing to speed up network training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTMGenerator(nn.Module):
    def __init__(self, input_dim, voc_dim, lstm_dim, embedding_dim, ngpu):
        """Constructor

        :param input_dim: (int) dimension of the input latent space vector
        :param voc_dim:
        :param lstm_dim:
        :param embedding_dim:
        :param ngpu: (int) number of GPUs to use for computation
        """
        # Call the superclass' constructor
        super(LSTMGenerator, self).__init__()

        self.ngpu = ngpu

        self.lstm_dim = lstm_dim
        self.embedding_dim = embedding_dim

        # Embedding layer which links a vocabulary to a latent vector space and vice versa
        self.embedding = nn.Embedding(voc_dim, embedding_dim)

        # LSTM layer with hidden states
        self.lstm = nn.LSTM(embedding_dim, lstm_dim, batch_first=True)

        # Final dense layer (corresponds to matrix multiplication)
        self.dense = nn.Linear(lstm_dim, voc_dim)
        # TODO: Soft-argmax operator instead of exact word predicted at step t-1

    def forward(self, input):
        return self.main(input)


def discriminator():
    pass
