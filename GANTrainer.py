import copy
import torch
from datetime import datetime, timedelta
import os
import pickle as pkl

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
    def __init__(self, gen, gpu=False):
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


class GANTrainer:
    def __init__(self, gen, dis, max_len=64, batch_size=16, lr=0.0002, n_rollout=16, gpu=False):
        self.gpu = gpu
        self.n_rollout = n_rollout
        self.G = gen
        self.D = dis
        self.max_len = max_len
        self.batch_size = batch_size

        self.optimizerG_pretrain = torch.optim.Adam(self.G.parameters(), lr=lr)
        self.optimizerG = torch.optim.Adam(self.G.parameters(), lr=lr)
        self.optimizerD = torch.optim.Adam(self.D.parameters(), lr=lr)

        self.pretrain_loss_G = []
        self.loss_G = []
        self.loss_D = []

    def pretrain_generator(self, pretrain_data, n_steps, word_0=0):
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
        reward_func = Rollout(self.G, self.gpu)
        G_loss_tot = 0

        for _ in range(n_steps):
            input, target = prepare_gen_data(self.G.sample(self.batch_size, self.batch_size, word_0=0),
                                             max_seq_len=self.max_len, gpu=self.gpu)
            rewards = reward_func.get_reward(target, self.n_rollout, self.D)
            adversarial_loss = self.G.PGLoss(input, target, rewards)

            # Optimizer step
            self.optimizerG.zero_grad()
            adversarial_loss.backward()
            self.optimizerG.step()

            G_loss_tot += adversarial_loss.item()

        return G_loss_tot

    def train(self, train_data, num_epochs, backup=False):
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
                    output = self.D(real_data)[:, 1].view(-1)

                    # Compute loss on real data batch
                    criterion = torch.nn.BCELoss()
                    lossD_real = criterion(output, label)

                    # Backward pass
                    lossD_real.backward()

                    ## Fake Data ##
                    # Generate fake sentences using G network
                    fake = self.G.sample(n_samples=real_data.size(0), batch_size=batch_size, word_0=0)
                    label.fill_(fake_label)

                    output = self.D(fake.detach())[:, 1].view(-1)
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

                # Create a pickle backup every 20 batches
                if backup and i % 20 == 0:
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
