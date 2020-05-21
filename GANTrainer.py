import copy
import torch
from datetime import datetime, timedelta
import os
import pickle as pkl

# Enables GPU computing to speed up network training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Conventions for fake and real data labels
real_label = 0.9
fake_label = 0.1

### Source: qsdf ###
class Rollout:
    def __init__(self, gen, gpu=False):
        """
        Constructor
        :param gen: (pytorch network) generator network
        :param gpu: (bool) specifies whether to use GPU computation to speed up training or not
        """
        self.gen = gen
        self.old_model = copy.deepcopy(gen)
        self.max_seq_len = gen.max_len
        self.vocab_size = gen.voc_dim
        self.gpu = gpu

    def rollout_mc_search(self, sentences, index):
        """
        Generate remaining tokens using a Monte Carlo search
        :param sentences: (tensor of size batch_size * max_seq_len)
        :param index: (int) index of the word from which to start filling out the sentences
        :return: (tensor of size batch_size * max_seq_len) input sentences who have been filled out from the word at
            index 'index'
        """
        batch_size = sentences.size(0)

        # get current state
        hidden = self.gen.init_hidden(batch_size)
        inp = sentences[:, :index]
        out, hidden = self.gen.forward(inp, hidden, require_hidden=True)
        out = out.view(batch_size, -1, self.vocab_size)[:, -1]

        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:, :index] = sentences[:, :index]

        if self.gpu:
            samples = samples.cuda()

        # Monte Carlo search
        for i in range(index, self.max_seq_len):
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data
            inp = out.view(-1)

            out, hidden = self.gen.forward(inp, hidden, require_hidden=True)

        return samples

    def get_reward(self, sentences, rollout_num, dis, current_k=0):
        """
        Compute reward using a Monte Carlo search
        :param sentences: (tensor of size batch_size * max_seq_len) sequence to compute a reward value for
        :param rollout_num: (int) number of rollout steps to perform
        :param dis: (pytorch network) discriminator network instance
        :param current_k: (int) current training gen
        :return: reward: (tensor of size batch_size * 1) list of rewards for each sentence in the batch
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

### Source ###

class GANTrainer:
    def __init__(self, gen, dis, preproc, max_len=64, batch_size=16, lr=0.0002, n_rollout=16, gpu=False):
        """
        Constructor
        :param gen: (pytorch network) generator network
        :param dis: (pytorch network) discriminator network
        :param preproc: (Preprocessor) preprocessor object used for preparing data for training
        :param max_len: (int) length of sequences
        :param batch_size: (int) size of batches used for training
        :param lr: (float) learning rate to use for both networks' optimizers
        :param n_rollout: (int) number of steps for the rollout function to train the generator
        :param gpu: (bool) specifies whether to use GPU computing or not
        """
        self.gpu = gpu
        self.n_rollout = n_rollout
        self.G = gen
        self.D = dis
        self.preproc = preproc
        self.max_len = max_len
        self.batch_size = batch_size

        # Initialize the optimizers
        self.optimizerG_pretrain = torch.optim.Adam(self.G.parameters(), lr=lr)
        self.optimizerG = torch.optim.Adam(self.G.parameters(), lr=lr)
        self.optimizerD = torch.optim.Adam(self.D.parameters(), lr=lr)

        # Initialize loss arrays
        self.pretrain_loss_G = []
        self.loss_G = []
        self.loss_D = []

    def pretrain_generator(self, pretrain_data, n_steps, word_0=0):
        """
        Pretrain generator network for a given number of steps
        :param pretrain_data: (dataloader) set of sequences for pre-training
        :param n_steps: (int) number of steps for pre-training
        :param word_0: (int) index of the starting word for training sequences
        """
        self.pretrain_loss_G = []
        criterion = torch.nn.NLLLoss()

        for step in range(n_steps):
            for i, data in enumerate(pretrain_data):
                input = torch.zeros(data.size()).long()
                input[:, 0] = word_0
                input[:, 1:] = data[:, :self.max_len - 1]
                if self.gpu:
                    input = input.cuda()

                hidden = self.G.init_hidden(data.size(0))
                output = self.G.forward(input, hidden)
                loss = criterion(output, input.view(-1))

                # Optimization step
                self.optimizerG_pretrain.zero_grad()
                loss.backward()
                self.optimizerG_pretrain.step()

                self.pretrain_loss_G.append(loss)

                # Print
                if i % 20:
                    print("[{0}/{1}] Pretrain-loss: {2:.3f}".format(step, n_steps, loss))

    def train_generator(self, n_steps):
        """
        Train generator for a number of steps
        :param n_steps: (int) number of training steps
        """
        reward_func = Rollout(self.G, self.gpu)
        G_loss_tot = 0

        for _ in range(n_steps):
            samples = self.G.sample(self.batch_size, self.batch_size)
            rewards = reward_func.get_reward(samples, self.n_rollout, self.D)
            adversarial_loss = self.G.PGLoss(samples, samples, rewards)

            # Optimizer step
            self.optimizerG.zero_grad()
            adversarial_loss.backward()
            self.optimizerG.step()

            G_loss_tot += adversarial_loss.item()

        return G_loss_tot

    def train(self, train_data, num_epochs, backup=False):
        """
        Train both the generator and discriminator network in a GAN fashion for a number of epochs
        :param train_data: (dataloader)
        :param num_epochs: (int) number of training epochs
        :param backup: (bool) specifies whether to save the model every few epochs or not
        :return: tuple of two lists, one for the loss of the generator and the other for the loss of the discriminator
            computed during training
        """
        self.loss_D = []
        self.loss_G = []
        start_time = datetime.now()

        for epoch in range(num_epochs):
            for i, data in enumerate(train_data, 0):

                if i % 5 == 0:
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
                    fake = self.G.sample(n_samples=real_data.size(0), batch_size=batch_size, word_0=-1)
                    label.fill_(fake_label)

                    output = self.D(fake.detach())[:, 0].view(-1)
                    lossD_fake = criterion(output, label)

                    lossD_fake.backward()
                    lossD = lossD_real + lossD_fake
                    self.optimizerD.step()

                    self.loss_D.append(lossD.item())


                ### Step 2 (G network): max log(D(G(z)))
                lossG = self.train_generator(n_steps=1)

                ### Append losses
                self.loss_G.append(lossG)

                # ETA calculation
                elapsed = datetime.now() - start_time
                batches_completed = (i+1) + epoch * len(train_data)
                remaining_batches = num_epochs * len(train_data) - batches_completed
                remaining_sec = elapsed.seconds * (remaining_batches / batches_completed)
                eta = timedelta(seconds=remaining_sec)

                print('[{0}/{1}][{2}/{3}]\tDis Loss: {4:.3f}\tGen Loss: {5:.3f}\tETA: {6}'.format(
                    epoch+1, num_epochs, i+1, len(train_data), lossD.item(), lossG, eta
                ))

                # Create a pickle backup at regular intervals
                if backup and i % 50 == 0:
                    current_time = datetime.now()
                    time_str = current_time.strftime('%Y-%m-%d_%H-%M')

                    with open(os.path.join('Saves', 'GANTrainer_' + time_str + '.pkl'), mode='wb') as fd:
                        pkl.dump(self, fd)

        # Save final model to file
        current_time = datetime.now()
        time_str = current_time.strftime('%Y-%m-%d_%H-%M')

        with open(os.path.join('Saves', 'GANTrainer-FINAL_' + time_str + '.pkl'), mode='wb') as fd:
            pkl.dump(self, fd)

        return self.loss_G, self.loss_D
