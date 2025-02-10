import torch
import torch.nn as nn


class Linear_BE(nn.Module):
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


class NonLinearBE(nn.Module):  # BatchEnsemble layer with non-linear transformations
    def __init__(self, dim_in: int, dim_out: int, k=32, init="uniform", amplitude_init=1.0, activation_RS=torch.tanh):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.k = k
        self.activation_RS = activation_RS  # Allow passing a custom activation function

        self.R = nn.Parameter(torch.Tensor(k, dim_in))
        self.W = nn.Parameter(torch.Tensor(dim_in, dim_out))
        self.S = nn.Parameter(torch.Tensor(k, dim_out))
        self.B = nn.Parameter(torch.Tensor(k, dim_out))

        if init == "uniform":
            nn.init.uniform_(self.R, -1, 1)
            nn.init.uniform_(self.S, -1, 1)

        elif init == "ones":
            nn.init.ones_(self.R)
            nn.init.normal_(self.S)

        elif init == "normal":
            nn.init.normal_(self.S)
            nn.init.normal_(self.R)

        elif init == "laplace":
            dist = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
            self.R.data = dist.sample((k, dim_in))[:, :, 0]
            self.S.data = dist.sample((k, dim_out))[:, :, 0]

        else:
            raise ValueError("init should be 'uniform', 'normal', 'ones' or 'laplace'")

        self.R.data *= amplitude_init
        self.S.data *= amplitude_init

        nn.init.normal_(self.W)
        nn.init.normal_(self.B)

    def forward(self, X):
        """
        Non-linear version of BatchEnsemble transformation.
        """
        # Element-wise multiplication of X and non-linearly transformed R
        transformed_R = self.activation_RS(self.R)
        output = torch.mul(X, transformed_R)  # (batch_size, k, dim_in)

        # Matrix multiplication with W
        output = torch.einsum("bki,io->bko", output, self.W)  # (batch_size, k, dim_out)

        # Apply non-linear transformation on S and multiply element-wise, add B
        transformed_S = self.activation_RS(self.S)
        output = output * transformed_S + self.B  # (batch_size, k, dim_out)

        return output


class PiecewiseLinearEncoding(nn.Module):
    # YuryGorishniy, IvanRubachev,andArtemBabenko (2022).
    # "On embeddings for numerical features in tabular deep learning." In NeurIPS
    # https://arxiv.org/abs/2203.05556

    def __init__(self, num_bins=10):
        """
        PLE module for encoding numerical features.

        Args:
        - num_bins (int): Number of bins T for piecewise encoding.
        """
        super().__init__()
        self.num_bins = num_bins
        self.bin_edges = None  # To be initialized during fit or passed as input

    def fit_bins(self, x):
        self.bin_edges = []
        num_features = x.shape[1]

        for i in range(num_features):
            feature_values = x[:, i]  # Extraire les valeurs pour la ième feature

            bins = torch.quantile(feature_values, torch.linspace(0, 1, self.num_bins))  # Diviser selon les quantiles
            self.bin_edges.append(bins)

        self.bin_edges = [bins.to(x.device) for bins in self.bin_edges]

    def forward(self, x):
        """
        x: shape (batch, d), où d est le nombre de features.
        Return : Tensor de forme (batch, d * num_bins)
        """
        if self.bin_edges is None:
            self.fit_bins(x)

        batch_size, num_features = x.shape

        # Stocker les résultats pour chaque feature
        encoded_features = []

        for i in range(num_features):
            feature_values = x[:, i]  # Extraire une feature (batch,)

            # Calculer les encodages PLE pour cette feature
            bins = self.bin_edges[i]  # Bins spécifiques à la ième feature
            ple_values = torch.zeros((batch_size, self.num_bins), device=x.device)

            for t in range(1, self.num_bins):
                bt_minus1 = bins[t - 1]
                bt = bins[t]

                # Cas 1 : Valeurs avant le bin
                ple_values[:, t] = torch.where(
                    (feature_values < bt_minus1) & (t > 0),
                    torch.tensor(0.0, device=x.device),
                    ple_values[:, t]
                )

                # Cas 2 : Valeurs après le bin
                ple_values[:, t] = torch.where(
                    (feature_values >= bt) & (t < self.num_bins - 1),
                    torch.tensor(1.0, device=x.device),
                    ple_values[:, t]
                )

                # Cas 3 : Encodage proportionnel
                ple_values[:, t] = torch.where(
                    (feature_values >= bt_minus1) & (feature_values < bt),
                    (feature_values - bt_minus1) / (bt - bt_minus1 + 1e-9),
                    ple_values[:, t]
                )

            encoded_features.append(ple_values)

        # Concaténer les encodages de toutes les features
        encoded_output = torch.cat(encoded_features, dim=1)  # (batch, d * num_bins)

        return encoded_output
