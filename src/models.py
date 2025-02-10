import torch
import torch.nn as nn

from modules import *


class MLP(nn.Module):
    """
    Simple MLP model
    """

    def __init__(self, in_features: int, hidden_sizes: list[int], out_features: int, dropout_rate=0):
        """
        :param int in_features: input dimension
        :param list[int] hidden_sizes: hidden layer sizes
        :param int out_features: output dimension
        :param float dropout_rate: dropout rate, defaults to 0
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        layer_sizes = [in_features] + hidden_sizes + [out_features]

        layers = [*[MLP_layer(layer_sizes[i], layer_sizes[i+1], dropout_rate) for i in range(len(layer_sizes)-1)],
                  nn.Linear(layer_sizes[-1], out_features)
                  ]

        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        return self.layers(X)


class EnsembleModel(nn.Module):
    """
    Global ensemble model that : 
    - takes batched input (batch, in_features)
    - clones it k times (batch, k, in_features)
    - passes it through a backbone (which model you want e.g TabM, MLPk, etc.) (batch, k, hidden_sizes[-1])
    - passes the output through k prediction heads, mean over heads (batch, out_features)
    """

    def __init__(self, backbone: nn.Module, in_features: int, hidden_sizes: int, out_features: int, k=32, dropout_rate=0.1, head_aggregation: {"mean", "conf", "none"}="mean", get_confidence=False):
        """
        :param nn.Module backbone: backbone used for the model
        :param int in_features: input dimension
        :param list[int] hidden_sizes: hidden layer sizes
        :param int out_features: output dimension
        :param int k: number of models, defaults to 32
        :param float dropout_rate: dropout rate, defaults to 0 
        :param str head_aggregation: type of aggeration for the k models' predictions. Can be "mean" average, based on prediction "conf"idence, or no aggregation ("none"), defaults to "mean"
        :param bool get_confidence: if set to True, the model outputs both the predictions and their respective confidences, defaults to False
        """
        super().__init__()

        self.backbone = backbone(in_features, hidden_sizes, k, dropout_rate)
        self.in_features = in_features
        self.k = k

        self.head_aggregation = head_aggregation
        self.get_confidence = get_confidence

        self.pred_heads = nn.ModuleList([nn.Linear(hidden_sizes[-1], out_features) for _ in range(k)])

    def forward(self, X: torch.Tensor):
        # clone X to shape (batch, k, dim)
        X = X.unsqueeze(1).repeat(1, self.k, 1)

        # pass through backbone
        X = self.backbone(X)

        # pass through prediction heads
        preds = [head(X[:, i]) for i, head in enumerate(self.pred_heads)]

        # concatenate head predictions
        preds = torch.stack(preds, dim=1)

        # Compute confidence if needed
        if self.head_aggregation == "conf":
            # Binary classification -> multi-class classification (class 0, class 1)
            if preds.size(2) == 1:
                preds_conf = torch.cat((1 - torch.sigmoid(preds), torch.sigmoid(preds)), dim=2)
            else:
                preds_conf = preds

            preds_softmax = torch.softmax(preds_conf, dim=2)

            # prevent log(0) without in-place operation
            preds_softmax = torch.where(preds_softmax == 0, torch.tensor(1e-10, device=preds_softmax.device), preds_softmax)

            # compute entropy
            entropy = -torch.sum(preds_softmax * torch.log(preds_softmax), dim=2)
            max_possible_entropy = torch.log(torch.tensor(preds_softmax.size(2), device=preds_softmax.device))

            confidences = (max_possible_entropy - entropy) / max_possible_entropy  # 0 = no confidence, 1 = full confidence

            # Global confidence = aggregation of all confidences to show how confident the model is on the predictions
            match self.head_aggregation:
                case "mean":
                    global_confidence = confidences.mean(dim=1)
                case "conf":
                    global_confidence = (torch.softmax(confidences, dim=1) * confidences).sum(dim=1)
                case "none":
                    global_confidence = confidences

            # Convert nan to 0
            global_confidence = torch.nan_to_num(global_confidence)

        match self.head_aggregation:
            case "mean":
                preds = preds.mean(dim=1)
            case "conf":
                weights = torch.softmax(confidences, dim=1)
                # Preds shape: (batch, k, out_features)
                # Multiply each head prediction by its weight (= confidence)
                preds = (preds * weights.unsqueeze(2)).sum(dim=1)
                # if binary_classification:
                #     preds = preds[:, 1].unsqueeze(1)

        if self.get_confidence:
            return preds, global_confidence
        return preds


class TabMWithAttention(nn.Module):
    def __init__(self, in_features, hidden_sizes, embed_dim, output_dim=1, num_heads=4, k=32, dropout_rate=0.1):
        """
        Initialize the TabMWithAttention model.

        Args:
            in_features (int): Number of input features.
            hidden_sizes (list of int): Sizes of the hidden layers.
            embed_dim (int): Dimensionality of the embedding space.
            output_dim (int): Dimensionality of the output space. Default is 1.
            num_heads (int): Number of attention heads. Default is 4.
            k (int): Number of ensemble components for BatchEnsemble layers. Default is 32.
            dropout_rate (float): Dropout rate for regularization. Default is 0.1.
        """
        super().__init__()
        self.k = k
        self.feature_embedding = nn.Linear(in_features, embed_dim)

        # Attention Blocks
        self.attention_layers = nn.Sequential(
            *[AttentionBlock(embed_dim, num_heads, dropout_rate) for _ in range(len(hidden_sizes))]
        )

        # BatchEnsemble Layers
        layer_sizes = [embed_dim] + hidden_sizes
        self.tabm_layers = nn.Sequential(
            *[Linear_BE(layer_sizes[i], layer_sizes[i + 1], k, dropout_rate, initialize_to_1=True) for i in range(len(layer_sizes) - 1)]
        )

        self.output_layer = nn.Linear(hidden_sizes[-1], output_dim)  # Aggregate predictions for k outputs

    def forward(self, X):
        """
        Forward pass for the TabMWithAttention model.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        X = self.feature_embedding(X)  # in: (batch_size, num_features), out: (batch_size, embed_dim)
        X = X.unsqueeze(1).transpose(0, 1)  # out: (seq_len, batch_size, embed_dim), assuming seq_len = 1 for each instance

        # Attention Layers
        X = self.attention_layers(X)  # out: (seq_len, batch_size, embed_dim)
        X = X.transpose(0, 1).mean(dim=1)  # Average over seq_len: (batch_size, embed_dim)
        X = X.unsqueeze(1).repeat(1, self.k, 1)  # Reshape for TabM layers, out: (batch_size, k, embed_dim)

        # TabM layers
        X = self.tabm_layers(X)  # out: (batch_size, k, hidden_size[-1])
        X = self.output_layer(X)  # out: (batch_size, k, 1)
        X = X.mean(dim=1)  # Aggregate k outputs, out: (batch_size, 1)

        return X


class NonLinearTabM(nn.Module):
    def __init__(self, layers_shapes: list, k=32, mean_over_heads=True, init="uniform", amplitude=1.0, intermediaire=False, activationRS=torch.tanh):
        super().__init__()

        self.k = k
        self.intermediaire = intermediaire

        self.layers = torch.nn.ModuleList([NonLinearBE(layers_shapes[0], layers_shapes[1], k, init="ones", activation_RS=activationRS),
                                          torch.nn.ReLU(),
                                          torch.nn.Dropout(0.1)])

        for i in range(1, len(layers_shapes)-2):
            self.layers.append(NonLinearBE(layers_shapes[i], layers_shapes[i+1], k, init, amplitude, activation_RS=activationRS))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(0.1))

        self.pred_heads = nn.ModuleList([nn.Linear(layers_shapes[-2], layers_shapes[-1]) for _ in range(k)])
        self.mean_over_heads = mean_over_heads

    def forward(self, x):
        X = x.unsqueeze(1).repeat(1, self.k, 1)

        intermediaire = []

        for layer in self.layers:
            X = layer(X)

            if (isinstance(layer, NonLinearBE) or isinstance(layer, nn.Linear)) and self.intermediaire:
                intermediaire.append(X)

        if intermediaire:
            return intermediaire

        # predictions
        preds = torch.stack([head(X[:, i]) for i, head in enumerate(self.pred_heads)], dim=1)

        if self.mean_over_heads:
            return preds.mean(dim=1)
        return preds


class TabM_with_PLE(nn.Module):
    def __init__(self, layers_shapes, k=32, mean_over_heads=True, init="uniform", amplitude=1.0, num_bins=10):
        super().__init__()

        self.k = k

        # Module PLE pour encoder les features numériques
        self.ple = PiecewiseLinearEncoding(num_bins=num_bins)

        # Reste du modèle TabM
        self.layers = nn.ModuleList([Linear_BE(layers_shapes[0], layers_shapes[1], k, init="ones"),
                                     nn.ReLU(),
                                     nn.Dropout(0.1)])

        for i in range(1, len(layers_shapes) - 2):
            self.layers.append(Linear_BE(layers_shapes[i], layers_shapes[i + 1], k, init, amplitude))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.1))

        self.pred_heads = nn.ModuleList([nn.Linear(layers_shapes[-2], layers_shapes[-1]) for _ in range(k)])
        self.mean_over_heads = mean_over_heads

    def fit_bins(self, x):
        """Appelle fit_bins pour initialiser les bins du PLE."""
        self.ple.fit_bins(x)

    def forward(self, x):
        # Encodage PLE
        x_encoded = self.ple(x)

        # Réplication des inputs encodés pour chaque "head"
        X = x_encoded.unsqueeze(1).repeat(1, self.k, 1)

        # Forward pass à travers les couches du modèle
        for layer in self.layers:
            X = layer(X)

        # Prédictions
        preds = torch.stack([head(X[:, i]) for i, head in enumerate(self.pred_heads)], dim=1)
        if self.mean_over_heads:
            return preds.mean(dim=1)
        return preds
