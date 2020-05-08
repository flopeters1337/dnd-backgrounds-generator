import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import copy

# Enables GPU computing to speed up network training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Conventions for fake and real data labels
real_label = 1
fake_label = 0


def prepare_gen_data(samples, max_seq_len, start_word=0, gpu=False):
    inp = torch.zeros(samples.size()).long()
    target = samples
    inp[:, 0] = start_word
    inp[:, 1:] = target[:, :max_seq_len - 1]

    if gpu:
        return inp.cuda(), target.cuda()
    return inp, target


class Rollout:
    def __init__(self, gen, gpu=True):
        self.gen = gen
        self.old_model = copy.deepcopy(gen)
        self.max_seq_len = gen.max_len
        self.vocab_size = gen.voc_dim
        self.gpu = gpu

    def rollout_mc_search(self, sentences, given_num):
        """
        fill up remain tokens with MC search
        :param sentences: size of batch_size * max_seq_len
        :param given_num:
        :return:
        """
        batch_size = sentences.size(0)

        # get current state
        hidden = self.gen.init_hidden(batch_size)
        inp = sentences[:, :given_num]
        out, hidden = self.gen.forward(inp, hidden, require_hidden=True)
        out = out.view(batch_size, -1, self.vocab_size)[:, -1]

        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:, :given_num] = sentences[:, :given_num]

        if self.gpu:
            samples = samples.cuda()

        # MC search
        for i in range(given_num, self.max_seq_len):
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data
            inp = out.view(-1)

            out, hidden = self.gen.forward(inp, hidden, require_hidden=True)

        return samples

    def get_reward(self, sentences, rollout_num, dis, current_k=0):
        """
        get reward via Monte Carlo search
        :param sentences: size of batch_size * max_seq_len
        :param rollout_num:
        :param dis:
        :param current_k: current training gen
        :return: reward: [batch_size]
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num * self.max_seq_len, batch_size]).float()
            if self.gpu:
                rewards = rewards.cuda()
            idx = 0
            for i in range(rollout_num):
                for given_num in range(1, self.max_seq_len + 1):
                    samples = self.rollout_mc_search(sentences, given_num)
                    out = dis.forward(samples)
                    out = torch.nn.functional.softmax(out, dim=-1)
                    reward = out[:, current_k + 1]
                    rewards[idx] = reward
                    idx += 1

        rewards = torch.mean(rewards.view(batch_size, self.max_seq_len, rollout_num), dim=-1)
        return rewards


class LSTMGenerator(nn.Module):
    def __init__(self, voc_dim, lstm_dim, embedding_dim, max_len, gpu=False):
        """
        Constructor
        :param voc_dim: (int) vocabulary set's size
        :param lstm_dim: (int) size of hidden states for the lstm layer
        :param embedding_dim: (int) dimension of embedding vector
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
        self.softmax = nn.Softmax(dim=-1)  # -1 to infer the correct dimension

        self.init_params()

    def forward(self, input, hidden, require_hidden=False):
        """

        :param input:
        :param hidden:
        :param require_hidden:
        :return:
        """
        # Transform the input sequences into sequences of latent vectors
        embedding = self.embedding(input)

        if len(input.size()) == 1:  # Particular case when the sequences contain only one element
            embedding = embedding.unsqueeze(1)

        output, hidden = self.lstm(embedding, hidden)
        output = output.contiguous().view(-1, self.lstm_dim)  # Again, -1 to infer the correct dimension
        output = self.dense(output)
        prediction = self.softmax(output)

        if require_hidden:  # Used when sampling the network
            return prediction, hidden
        else:
            return prediction

    def sample(self, n_samples, batch_size, word_0, gen_type='multinom'):
        if n_samples != batch_size:
            n_batches = n_samples // batch_size + 1
        else:
            n_batches = 1

        samples = torch.zeros(n_batches * batch_size, self.max_len).long()

        # Produce samples by batches
        for batch in range(n_batches):
            hidden = self.init_hidden(batch_size)
            input = torch.LongTensor([word_0] * batch_size)  # Initialize every sequence with 'word_0' as starting token
            if self.gpu:
                input = input.cuda()

            # Iterate the generator until we reach the maximum length allowed for the sequence
            for i in range(self.max_len):
                # Forward pass where we keep track of the hidden states of the network
                output, hidden = self.forward(input, hidden, require_hidden=True)

                if gen_type == 'multinom':
                    # Generate the next token in the sequence randomly using the output as a multinomial distribution
                    next_token = torch.multinomial(output, 1)
                elif gen_type == 'argmax':
                    # Choose the most probable token in the sequence deterministically
                    next_token = torch.argmax(output, 1)

                samples[batch * batch_size:(batch + 1) * batch_size, i] = next_token.view(-1)
                input = next_token.view(-1)

        samples = samples[:n_samples]

        return samples

    def PGLoss(self, input, target, reward):
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
                torch.nn.init.normal_(param, mean=0, std=1)  # TODO: maybe change depending on the parameter?

    def init_hidden(self, batch_size):
        # Initialize all hidden states to 0
        h = torch.zeros(1, batch_size, self.lstm_dim)
        c = torch.zeros(1, batch_size, self.lstm_dim)

        if self.gpu:
            return h.cuda(), c.cuda()
        else:
            return h, c


class GANTrainer:
    def __init__(self, gen, dis, max_len=64, batch_size=16, lr=0.0002, n_rollout=16, gpu=False):
        self.gpu = gpu
        self.n_rollout = n_rollout
        self.G = gen
        self.D = dis
        self.max_len = max_len
        self.batch_size = batch_size

        # Build mappings from word to indices and vice-versa
        #self.word2idx = {vocab[i]: i for i in range(len(vocab))}
        #self.idx2word = {v: k for k, v in self.word2idx.items()}

        self.optimizerG = torch.optim.Adam(self.G.parameters(), lr=lr)
        self.optimizerD = torch.optim.Adam(self.D.parameters(), lr=lr)

        self.loss_G = []
        self.loss_D = []

    def train_generator(self, n_steps):
        reward_func = Rollout(self.G, self.gpu)
        G_loss_tot = 0

        for _ in range(n_steps):
            input, target = prepare_gen_data(self.G.sample(self.batch_size, self.batch_size, word_0=0), max_seq_len=self.max_len,
                                             gpu=self.gpu)
            rewards = reward_func.get_reward(target, self.n_rollout, self.D)
            adversarial_loss = self.G.PGLoss(input, target, rewards)

            # Optimizer step
            self.optimizerG.zero_grad()
            adversarial_loss.backward()
            self.optimizerG.step()

            G_loss_tot += adversarial_loss.item()

    def train(self, train_data, num_epochs):
        for epoch in range(num_epochs):
            for i, data in enumerate(train_data, 0):
                ### Step 1 (D network): max log(D(x)) + log(1 - D(G(z))) ###

                ## Real Data ##
                self.D.zero_grad()
                real_data = data.to(device)
                batch_size = real_data.size(0)
                label = torch.full((batch_size,), real_label, device=device)

                # Forward pass
                output = self.D(real_data)[:, 0].view(-1)

                # Compute loss on real data batch
                criterion = torch.nn.BCELoss()
                lossD_real = criterion(output, label)

                # Backward pass
                lossD_real.backward()

                ## Fake Data ##
                # Generate fake sentences using G network
                fake = self.G.sample(n_samples=real_data.size(0), batch_size=batch_size, word_0=0)
                label.fill_(fake_label)

                output = self.D(fake.detach())[:, 0].view(-1)
                lossD_fake = criterion(output, label)

                lossD_fake.backward()
                lossD = lossD_real + lossD_fake
                self.optimizerD.step()

                ### Step 2 (G network): max log(D(G(z)))
                self.train_generator(n_steps=1)
