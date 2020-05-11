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
FORCE_PREPROCESS = False
BATCH_SIZE = 128

if GPU:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


if __name__ == '__main__':
    if DEBUG:
        #data = [torch.randint(0, 7, size=(10, 8))]
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

        dis = CNNDiscriminator(batch_size=1, max_seq=3, voc_size=8)
        gen = LSTMGenerator(voc_dim=8, lstm_dim=4, embedding_dim=4, max_len=3)

        trainer = GANTrainer(gen, dis, max_len=3, batch_size=1)
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
            data, _ = preproc.preprocess() # Only pick sentences
            with open('cached_data.pkl', mode='wb') as fd:
                pkl.dump((data, preproc), fd)

        dataset = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

        dis = CNNDiscriminator(batch_size=BATCH_SIZE, max_seq=40, voc_size=len(preproc.vocabulary)+1)
        gen = LSTMGenerator(voc_dim=len(preproc.vocabulary)+1, lstm_dim=16, embedding_dim=16, max_len=40, gpu=GPU)

        print('Training...')
        trainer = GANTrainer(gen, dis, max_len=40, batch_size=BATCH_SIZE, n_rollout=4, backup=True, gpu=GPU)
        lossG, lossD = trainer.train(dataset, 5)

    with open('losses.pkl', mode='wb') as fd:
        pkl.dump((lossG, lossD), fd)

    # Create graph
    sns.set()
    sns.lineplot(x=range(len(lossG)), y=lossG)
    ax = sns.lineplot(x=range(len(lossD)), y=lossD)
    ax.set(xlabel='# of epochs', ylabel='Loss')
    ax.legend(['Generator', 'Discriminator'])
    plt.show()
