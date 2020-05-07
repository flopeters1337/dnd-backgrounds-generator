import torch
from CNNDiscriminator import CNNDiscriminator
from dndGAN import LSTMGenerator, GANTrainer


if __name__ == '__main__':
    data = [torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            torch.tensor([7, 6, 5, 4, 3, 2, 1, 0])]

    dis = CNNDiscriminator(batch_size=1, max_seq=8, voc_size=8)
    gen = LSTMGenerator(voc_dim=8, lstm_dim=4, embedding_dim=4, max_len=8)

    trainer = GANTrainer(gen, dis, max_len=8, batch_size=1)
    trainer.train(data, 2)
