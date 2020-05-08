import torch
from CNNDiscriminator import CNNDiscriminator
from dndGAN import LSTMGenerator, GANTrainer


if __name__ == '__main__':
    data = [torch.randint(0, 7, size=(10, 8))]

    dis = CNNDiscriminator(batch_size=1, max_seq=8, voc_size=8)
    gen = LSTMGenerator(voc_dim=8, lstm_dim=4, embedding_dim=4, max_len=8)

    trainer = GANTrainer(gen, dis, max_len=8, batch_size=1)
    trainer.train(data, 2)
