
# Implementation of the discriminator neural network (CNN architecture), inheriting from nn.Module super class

import torch
import torch.nn as nn

class CNNDiscriminator(nn.Module):
    
    def __init__(self, batch_size=8, max_seq=5, voc_size=50, embedding_dim=10, 
                 window_sizes=[1,2,3], n_filters = [300,300,300], n_inter_nodes=200):
        """
        Constructor
        :param batch_size: (int) 
        :param max_seq: (int) 
        :param voc_size: (int)
        :param embedding_dim: (int)
        :param windows_size: (list of int)
        :param n_filters: (list of int)
        :param n_inter_nodes: (int) 
        """
      
        # Call the superclass' (nn.Module) constructor
        super(CNNDiscriminator, self).__init__()
        
        # Embedding layer
        # NB : Why padding in the TextGan version ?
        self.embedding = nn.Embedding(voc_size, embedding_dim) 
        
        # Tanh layer
        self.tan_layer = nn.Tanh()
        
        # Convolutional layers with multiples filters with different sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=f, kernel_size=(h, embedding_dim), stride=1, padding=0)
            for h,f in zip(window_sizes, n_filters)
        ])
        
        # Number of features map = sum of filters = number of nodes in the first layer of the FC NN classifier
        self.n_features_maps = sum(n_filters)
     
        # Final classifier : a 2 layers fully connected nn with sigmoid activation function and softmax at the end
        # NB : Quid formule TextGan at the end ? 
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.n_features_maps, out_features=n_inter_nodes),
            nn.Sigmoid(),
            nn.Linear(in_features=n_inter_nodes, out_features=2),
            nn.Softmax(dim=1)) 
        
        # Call the parameters initialization function
        self.init_params() 
    
    def perform_convolution(self, x):
        
        """
        Performing the multiple convolutional filters with tanh + max pooling over time
        :param x: (tensor) of dimension batch_size * 1 * max_seq_len * embed_dim
        """
        
        # Apply convolutional layers + following tanh layers
        # NB : Why Relu in some implementations ? 
        outputs = [conv(x).squeeze(3) for conv in self.conv_layers] # outputs = batch_size * num_filter * length
        outputs = [self.tan_layer(output) for output in outputs] # outputs = batch_size * num_filter * length
        
        # Prepare and apply the max-pooling filters over time
        poolings = [nn.MaxPool1d(output.size(2))for output in outputs]
        outputs = [pool(output).squeeze(2) for output, pool in zip(outputs, poolings)] # outputs = batch_size * n_filters
        
        # Concatane the features obtained with the different filter sizes
        output = torch.cat(outputs,1) # output = batch_size * feature_dim
        
        return output
        
    def forward(self, x):
        
        """
        Overwriting the base forward function in nn.Module and performing a forward pass trough the nn 
        :param x: (tensor) of dimension batch_size * max_seq_len 
        """
        
        # Retrieve the embedding of the words and unsqueezing for dimentional fitting for next layer
        output = self.embedding(x).unsqueeze(1) # output = batch_size * 1 * max_seq_len * embed_dim
        
        # Perform multiple convolution
        output = self.perform_convolution(output) # output = batch_size * feature_dim
        
        # Return the final output (two probabilities (sum = 1) of being a fake and a real samples
        output = self.classifier(output) # batch_size * 2
        
        return output 
    
    def init_params(self):
        
        """
        Initiate the parameters to a normal distribution with a mean of 0 and a standard deviation of 1 
        """
        for param in self.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0, std=1)
