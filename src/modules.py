import torch
import torch.nn as nn


class linear_BE(nn.Module):
    """
    Linear layer with Batch Embedding
    """

    def __init__(self, in_features: int, out_features: int, k=32, dropout_rate=0, initialize_to_1=False):
        """
        :param int in_features: input dimension
        :param int out_features: output dimension
        :param int k: number of models, defaults to 32
        :param float dropout_rate: dropout rate, defaults to 0
        :param bool initialize_to_1: initialize R and S to 1, as needed for some backbones, defaults to False
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        if initialize_to_1:  # For TabM
            self.R = nn.Parameter(torch.ones(k, in_features))
            self.S = nn.Parameter(torch.ones(k, out_features))
        else:
            # Paper generates randomly with +-1
            self.R = nn.Parameter(torch.zeros((k, in_features)))
            nn.init.uniform_(self.R, -1, 1)
            self.S = nn.Parameter(torch.zeros((k, out_features)))
            nn.init.uniform_(self.S, -1, 1)

        self.W = nn.Parameter(torch.zeros((in_features, out_features)))
        nn.init.uniform_(self.W, -1, 1)
        self.B = nn.Parameter(torch.zeros((k, out_features)))
        nn.init.uniform_(self.B, -1, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X: torch.Tensor):
        """
        Shapes:

        X: (batch_size, k, in_features)
        R: (k, in_features)
        W: (in_features, out_features)
        S: (k, out_features)
        B: (k, out_features)
        output: (batch_size, k, out_features)

        Formula:
        output = ( (X * R) W) * S + B
        """
        output = X * self.R

        output = torch.einsum("bki,io->bko", output, self.W)
        output = output * self.S + self.B
        output = self.relu(output)
        output = self.dropout(output)

        return output

    def extra_repr(self):
        """
        Adds information about the layer to its string representation (useful when printing the model)
        """
        return f"in_features={self.in_features}, out_features={self.out_features}"


class MLP_layer(nn.Module):
    """
    MLP layer with Linear, ReLU and Dropout
    """

    def __init__(self, in_features: int, out_features: int, dropout_rate=0):
        """
        :param int in_features: input dimension
        :param int out_features: output dimension
        :param float dropout_rate: dropout rate, defaults to 0
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X: torch.Tensor):
        output = self.linear(X)
        output = self.relu(output)
        output = self.dropout(output)

        return output

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"


class MLPk_layer(nn.Module):
    """
    k parallel MLP layers with Linear, ReLU and Dropout
    """

    def __init__(self, in_features: int, out_features: int, k=32, dropout_rate=0):
        """
        :param int in_features: input dimension
        :param int out_features: output dimension
        :param int k: number of models, defaults to 32
        :param float dropout_rate: dropout rate, defaults to 0
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros((k, in_features, out_features)))
        nn.init.uniform_(self.W, -1, 1)
        self.B = nn.Parameter(torch.zeros((k, out_features)))
        nn.init.uniform_(self.B, -1, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X: torch.Tensor):
        """
        Shapes:

        X: (batch_size, k, in_features)
        W: (k, in_features, out_features)
        B: (k, out_features)
        output: (batch_size, k, out_features)

        Formula:
        output = X @ W + B
        """
        output = torch.einsum("bki,kio->bko", X, self.W)
        output = output + self.B

        output = self.relu(output)
        output = self.dropout(output)

        return output

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0):
        """
        Initialize the AttentionBlock.

        Args:
            embed_dim (int): Dimensionality of the embedding space.
            num_heads (int): Number of attention heads.
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, X):
        """
        Forward pass for the AttentionBlock.

        Args:
            X (torch.Tensor): Input tensor of shape (seq_len, batch_size, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, embed_dim).
        """
        attn_output, _ = self.self_attn(X, X, X)
        X = X + self.dropout1(attn_output)  # Add residual connections
        X = self.norm1(X)

        ffn_output = self.ffn(X)
        X = X + self.dropout2(ffn_output)  # Add residual connections
        X = self.norm2(X)
        return X
