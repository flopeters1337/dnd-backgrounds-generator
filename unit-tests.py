import pickle as pkl
import torch
from CNNDiscriminator import CNNDiscriminator
from LSTMGenerator import LSTMGenerator
from GANTrainer import GANTrainer
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
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

    with open('losses.pkl', mode='wb') as fd:
        pkl.dump((lossG, lossD), fd)

    # Create graph
    sns.set()
    sns.lineplot(x=range(len(lossG)), y=lossG)
    ax = sns.lineplot(x=range(len(lossD)), y=lossD)
    ax.set(xlabel='# of epochs', ylabel='Loss')
    ax.legend(['Generator', 'Discriminator'])
    plt.show()
