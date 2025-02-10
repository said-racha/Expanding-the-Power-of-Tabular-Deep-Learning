import torch
import torch.nn as nn

from models import *


class TabM_naive(nn.Module):
    """
    TabM_naive backbone
    BatchEnsemble layers with separate prediction heads
    """

    def __init__(self, in_features: int, hidden_sizes: list[int], k=32, dropout_rate=0):
        """
        :param int in_features: input dimension
        :param list[int] hidden_sizes: hidden layer sizes
        :param int k: number of models, defaults to 32
        :param float dropout_rate: dropout rate, defaults to 0
        """
        super().__init__()

        self.in_features = in_features
        self.hidden_sizes = hidden_sizes
        self.k = k

        layer_sizes = [in_features] + hidden_sizes

        layers = [linear_BE(layer_sizes[i], layer_sizes[i+1], k, dropout_rate) for i in range(len(layer_sizes)-1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        return self.layers(X)


class TabM_mini(nn.Module):
    """
    TabM_mini backbone
    Shared layer weights except for the first R
    """

    def __init__(self, in_features: int, hidden_sizes: list[int], k=32, dropout_rate=0):
        """
        :param int in_features: input dimension
        :param list[int] hidden_sizes: hidden layer sizes
        :param int k: number of models, defaults to 32
        :param float dropout_rate: dropout rate, defaults to 0
        """
        super().__init__()

        self.k = k

        self.R = nn.Parameter(torch.randn(k, in_features))

        layer_sizes = [in_features] + hidden_sizes

        layers = [MLP_layer(layer_sizes[i], layer_sizes[i+1], dropout_rate) for i in range(len(layer_sizes)-1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        output = X * self.R
        return self.layers(output)


class TabM(nn.Module):
    """
    TabM backbone
    All layers have separate weights initialized to 1
    """

    def __init__(self, in_features: int, hidden_sizes: list[int], k=32, dropout_rate=0):
        """
        :param int in_features: input dimension
        :param list[int] hidden_sizes: hidden layer sizes
        :param int k: number of models, defaults to 32
        :param float dropout_rate: dropout rate, defaults to 0
        """
        super().__init__()

        self.k = k

        layer_sizes = [in_features] + hidden_sizes

        layers = [linear_BE(layer_sizes[i], layer_sizes[i+1], k, dropout_rate, initialize_to_1=True) for i in range(len(layer_sizes)-1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        return self.layers(X)


class MLPk(nn.Module):
    """
    MLPk backbone
    k parallel MLP layers
    """

    def __init__(self, in_features: int, hidden_sizes: list[int], k=32, dropout_rate=0):
        """
        :param int in_features: input dimension
        :param list[int] hidden_sizes: hidden layer sizes
        :param int k: number of models, defaults to 32
        :param float dropout_rate: dropout rate, defaults to 0
        """
        super().__init__()

        layer_sizes = [in_features] + hidden_sizes

        layers = [MLPk_layer(layer_sizes[i], layer_sizes[i+1], k, dropout_rate) for i in range(len(layer_sizes)-1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        return self.layers(X)
