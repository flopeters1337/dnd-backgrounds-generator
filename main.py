import os
import pickle as pkl
import torch
from torch.utils.data import DataLoader
from Preprocessor import Preprocessor
from CNNDiscriminator import CNNDiscriminator
from LSTMGenerator import LSTMGenerator
from GANTrainer import GANTrainer
import seaborn as sns
import matplotlib.pyplot as plt

GPU = True
DEBUG = False
FORCE_PREPROCESS = True
BATCH_SIZE = 128
SEQ_LEN = 25
EMBED_DIM = 32

if GPU:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


if __name__ == '__main__':

    # Toy examples for debugging
    if DEBUG:
        data = [torch.tensor([[0, 2, 0],
                             [0, 1, 0],
                             [0, 3, 0],
                             [0, 1, 0],
                             [0, 4, 0],
                             [0, 4, 0],
                             [0, 7, 0],
                             [0, 5, 0],
                             [0, 0, 0],
                             [0, 3, 0]])]

        dis = CNNDiscriminator(batch_size=1, max_seq=4, voc_size=8)
        gen = LSTMGenerator(voc_dim=8, lstm_dim=4, embedding_dim=4, max_len=4, gpu=GPU)

        trainer = GANTrainer(gen, dis, max_len=4, batch_size=1, gpu=GPU)
        lossG, lossD = trainer.train(data, 150)
    else:
        preproc = Preprocessor()
        print('Loading dataset...')
        preproc.load_dataset(os.path.join('Data', 'dd_bios.xls'))

        print('Preprocessing dataset...')
        if os.path.exists('cached_data.pkl') and not FORCE_PREPROCESS:
            with open('cached_data.pkl', mode='rb') as fd:
                data, preproc = pkl.load(fd)
        else:
            data, _ = preproc.preprocess(max_sentences=SEQ_LEN)  # Only pick sentences

            # Save data and preprocessor to not perform a useless preprocessing step
            with open('cached_data.pkl', mode='wb') as fd:
                pkl.dump((data, preproc), fd)

        dataset = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

        dis = CNNDiscriminator(window_sizes=[3, 4, 5], embedding_dim=EMBED_DIM, batch_size=BATCH_SIZE, max_seq=SEQ_LEN,
                               voc_size=len(preproc.vocabulary)+1)
        gen = LSTMGenerator(voc_dim=len(preproc.vocabulary)+1, lstm_dim=500, embedding_dim=EMBED_DIM,
                            max_len=SEQ_LEN, gpu=GPU)

        print('Training...')
        trainer = GANTrainer(gen, dis, preproc, max_len=SEQ_LEN, batch_size=BATCH_SIZE, n_rollout=15, lr=0.005, gpu=GPU)
        trainer.pretrain_generator(dataset, 5)
        lossG, lossD = trainer.train(dataset, num_epochs=10, backup=True)

    with open('losses.pkl', mode='wb') as fd:
        pkl.dump((lossG, lossD), fd)

    # Plot losses
    sns.set()
    sns.lineplot(x=range(len(lossG)), y=lossG)
    ax = sns.lineplot(x=range(len(lossD)), y=lossD)
    ax.set(xlabel='# of batches', ylabel='Loss')
    ax.legend(['Generator', 'Discriminator'])
    plt.show()
