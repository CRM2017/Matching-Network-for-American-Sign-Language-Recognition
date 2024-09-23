import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from math import sqrt


class CNNEncoder(nn.Module):
    """Encoder for feature embedding"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        finalSize = int(math.floor(80 / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * 64

        def forward(self, x):
            """x: bs*3*84*84 """
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

            return out


def cosine_distance(support_set, input_image):
    """
    Compute cosine distance between support set and input image
    :param support_set:the embeddings of the support set images.shape[sequence_length,batch_size,64]
    :param input_image: the embedding of the target image,shape[batch_size,64]
    :return:shape[batch_size,sequence_length]
    """
    similarities = []
    for support_image in support_set:
        cos = nn.CosineSimilarity(dim=1, eps=1e-10)
        cosine_similarity = cos(support_image, input_image)  # Compute cosine distance
        similarities.append(cosine_similarity)
    similarities = torch.stack(similarities)
    return similarities.t()


def pairwise_distance(support_set, input_image):
    """
    Compute pairwise distance between support set and input image
    :param support_set:the embeddings of the support set images.shape[sequence_length,batch_size,64]
    :param input_image: the embedding of the target image,shape[batch_size,64]
    :return:shape[batch_size,sequence_length]
    """
    similarities = []
    for support_image in support_set:
        pdist = nn.PairwiseDistance(p=2)
        pairwise_dist = pdist(support_image, input_image)
        similarities.append(pairwise_dist)
    similarities = torch.stack(similarities)
    return similarities.t()


def euclidean_distance(support_set, input_image):
    similarities = []
    for support_image in support_set:
        euclidean_dist = torch.cdist(support_image, input_image, p=2)
        print(euclidean_dist.shape)
        similarities.append(euclidean_dist)
    similarities = torch.stack(similarities)
    return similarities.t()


def attention_kernel (similarities, support_set_y):
    """
    Products pdfs over the support set classes for the target set image.
    :param similarities: A tensor with cosine similarities of size[batch_size,sequence_length]
    :param support_set_y:[batch_size,sequence_length,classes_num]
    :return: Softmax pdf shape[batch_size,classes_num]
    """
    softmax = nn.Softmax()
    softmax_similarities = softmax(similarities)
    preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
    return preds


def convLayer(in_channels, out_channels, dropout=0.0):
    """3*3 convolution with padding,ever time call it the output size become half"""
    cnn_seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(dropout)
    )
    return cnn_seq

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 16)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, layer_size=64, num_channels=1, dropout=1.0, image_size=28):
        super(CNN, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param dropout:
        :param image_size:
        """
        self.layer1 = convLayer(num_channels, layer_size, dropout)
        self.layer2 = convLayer(layer_size, layer_size, dropout)
        self.layer3 = convLayer(layer_size, layer_size, dropout)
        self.layer4 = convLayer(layer_size, layer_size, dropout)

        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size

    def forward(self, image_input):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size()[0], -1)
        return x


class AttentionKernel(nn.Module):
    def __init__(self):
        super(AttentionKernel, self).__init__()

    def forward(self, similarities, support_set_y):
        """
        Products pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarites of size[batch_size,sequence_length]
        :param support_set_y:[batch_size,sequence_length,classes_num]
        :return: Softmax pdf shape[batch_size,classes_num]
        """
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities)
        preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
        return preds


class BiLSTM(nn.Module):
    def __init__(self, layer_size, batch_size, vector_dim,use_cuda):
        super(BiLSTM, self).__init__()
        """
        Initial a muti-layer Bidirectional LSTM
        :param layer_size: a list of each layer'size
        :param batch_size: 
        :param vector_dim: 
        """
        self.batch_size = batch_size
        self.hidden_size = layer_size[0]
        self.vector_dim = vector_dim
        self.num_layer = len(layer_size)
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(input_size=self.vector_dim, num_layers=self.num_layer, hidden_size=self.hidden_size,
                            bidirectional=True)
        self.hidden = self.init_hidden(self.use_cuda)

    def init_hidden(self, use_cuda):
        if use_cuda:
            return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda(),
                    Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda())
        else:
            return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False),
                    Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False))

    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""

        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, inputs):
        self.hidden = self.init_hidden(self.use_cuda)
        # self.hidden = self.repackage_hidden(self.hidden)
        output, self.hidden = self.lstm(inputs, self.hidden)
        return output


class MatchingNetwork(nn.Module):
    def __init__(self, dropout, batch_size=32, num_channels=1, learning_rate=1e-3, fce=False, num_classes_per_set=5, \
                 num_samples_per_class=1, image_size=28, use_cuda=True):
        """
        This is our main network
        :param dropout: dropout rate
        :param batch_size:
        :param num_channels:
        :param learning_rate:
        :param fce: Flag indicating whether to use full context embeddings(i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set:
        :param num_samples_per_class:
        :param image_size:
        """
        super(MatchingNetwork, self).__init__()
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.fce = fce
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.image_size = image_size
        self.g = CNN(layer_size=64, num_channels=num_channels, dropout=dropout, image_size=image_size)
        self.attention = AttentionKernel()
        if self.fce:
            self.lstm = BiLSTM(layer_size=[32], batch_size=self.batch_size, vector_dim=self.g.outSize, use_cuda=use_cuda)

    def forward(self, support_set_images, support_set_y_one_hot, target_image, target_y):
        """
        Main process of the network
        :param support_set_images: shape[batch_size,sequence_length,num_channels,image_size,image_size]
        :param support_set_y_one_hot: shape[batch_size,sequence_length,num_classes_per_set]
        :param target_image: shape[batch_size,num_channels,image_size,image_size]
        :param target_y:
        :return:
        """
        # produce embeddings for support set images
        encoded_images = []
        for i in np.arange(support_set_images.size(1)):
            gen_encode = self.g(support_set_images[:, i, :, :])
            encoded_images.append(gen_encode)

        # produce embeddings for target images
        gen_encode = self.g(target_image)
        encoded_images.append(gen_encode)
        output = torch.stack(encoded_images)

        # use fce?
        if self.fce:
            outputs = self.lstm(output)

        # Compute similarities between support set embeddings and query
        similarities = cosine_distance(support_set=output[:-1], input_image=output[-1])


        preds = attention_kernel(similarities, support_set_y=support_set_y_one_hot)

        # calculate the accuracy
        values, indices = preds.max(1)
        accuracy = torch.mean((indices.squeeze() == target_y).float())
        crossentropy_loss = F.cross_entropy(preds, target_y.long())

        return accuracy, crossentropy_loss
