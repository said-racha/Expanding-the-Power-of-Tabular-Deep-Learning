import torch
import torch.nn as nn

from copy import deepcopy


# ===== LAYERS =====


class LinearBE(nn.Module): # BatchEnsemble layer
    def __init__(self, dim_in:int, dim_out:int, k=32, init="uniform", amplitude_init=1.0):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.k = k
        self.R = nn.Parameter(torch.Tensor(k, dim_in))
        self.W = nn.Parameter(torch.Tensor(dim_in, dim_out))
        self.S = nn.Parameter(torch.Tensor(k, dim_out))
        self.B = nn.Parameter(torch.Tensor(k, dim_out))

        if init == "uniform": # TabM naive
            # randomly initialized with ±1 to ensure diversity of the k linear layers
            nn.init.uniform_(self.R, -1, 1)
            nn.init.uniform_(self.S, -1, 1)

        elif init == "ones": # TabM
            nn.init.ones_(self.R)
            nn.init.normal_(self.S)
        
        elif init == "normal":
            nn.init.normal_(self.S)
            nn.init.normal_(self.R)

        elif init == "laplace":
            dist = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
            self.R.data = dist.sample((k, dim_in))[:,:,0]
            self.S.data = dist.sample((k, dim_out))[:,:,0]
        
        else:
            raise ValueError("init should be 'uniform', 'normal', 'ones' or 'laplace'")

        self.R.data *= amplitude_init
        self.S.data *= amplitude_init

        nn.init.normal_(self.W)
        nn.init.normal_(self.B)


    def forward(self, X):
        """
        LinearBE(X) = ( (X * R) W) * S + B

        X has shape (batch_size, k, dim_in)
        R has shape (k, dim_in)
        W has shape (dim_in, dim_out)
        S has shape (k, dim_out)
        B has shape (k, dim_out)
        """

        # Element-wise multiplication of X and R
        output = torch.mul(X, self.R) # (batch_size, k, dim_in)

        # Matrix multiplication with W
        output = torch.einsum("bki,io->bko", output, self.W)  # (batch_size, k, dim_out)

        # Element-wise multiplication with S and addition of B
        output = output * self.S + self.B  # (batch_size, k, dim_out)

        return output
    


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
            
            bins = torch.quantile(feature_values, torch.linspace(0, 1, self.num_bins))# Diviser selon les quantiles
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

            for t in range(1,self.num_bins):
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


    

# ===== BACKBONES =====

class TabM_Naive(nn.Module):
    def __init__(self, layers_shapes:list, k=32, mean_over_heads = True, init="uniform", amplitude=1.0, intermediaire=False):
        super().__init__()

        self.k = k
        self.intermediaire = intermediaire

        self.layers = torch.nn.ModuleList()

        # applying BatchEnsemble to all linear layers
        for i in range(len(layers_shapes)-2):
            self.layers.append(LinearBE(layers_shapes[i], layers_shapes[i+1], k, init, amplitude))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(0.1))
        
        # fully non-shared prediction heads
        self.pred_heads = nn.ModuleList([nn.Linear(layers_shapes[-2], layers_shapes[-1]) for _ in range(k)])
        self.mean_over_heads = mean_over_heads

    
    def forward(self, x): 
        """
        X represents k representations of the same input object x
        so, x has shape (batch, dim)
        and X has shape (batch, k, dim)
        """
        X = x.unsqueeze(1).repeat(1, self.k, 1)

        intermediaire = []

        for layer in self.layers:
            X = layer(X)

            if (isinstance(layer, LinearBE) or isinstance(layer, nn.Linear)) and self.intermediaire:
                intermediaire.append(X)
        
        if intermediaire:
            return intermediaire
        
        # predictions
        preds = torch.stack([head(X[:,i]) for i, head in enumerate(self.pred_heads)], dim=1)
        
        if self.mean_over_heads:
            return preds.mean(dim=1)
        return preds


class TabM(nn.Module):
    def __init__(self, layers_shapes:list, k=32, mean_over_heads = True, init="uniform", amplitude=1.0, intermediaire=False):
        super().__init__()

        self.k = k
        self.intermediaire = intermediaire

        self.layers = torch.nn.ModuleList([LinearBE(layers_shapes[0], layers_shapes[1], k, init="ones"),
                                          torch.nn.ReLU(),
                                          torch.nn.Dropout(0.1)])

        for i in range(1,len(layers_shapes)-2):
            self.layers.append(LinearBE(layers_shapes[i], layers_shapes[i+1], k, init, amplitude))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(0.1))
        
        self.pred_heads = nn.ModuleList([nn.Linear(layers_shapes[-2], layers_shapes[-1]) for _ in range(k)])
        self.mean_over_heads = mean_over_heads

    
    def forward(self, x):
        X = x.unsqueeze(1).repeat(1, self.k, 1)

        intermediaire = []

        for layer in self.layers:
            X = layer(X)

            if (isinstance(layer, LinearBE) or isinstance(layer, nn.Linear)) and self.intermediaire:
                intermediaire.append(X)
        
        if intermediaire:
            return intermediaire
        
        # predictions
        preds = torch.stack([head(X[:,i]) for i, head in enumerate(self.pred_heads)], dim=1)
        
        if self.mean_over_heads:
            return preds.mean(dim=1)
        return preds

class NonLinearTabM(nn.Module):
    def __init__(self, layers_shapes:list, k=32, mean_over_heads = True, init="uniform", amplitude=1.0, intermediaire=False, activationRS=torch.tanh):
        super().__init__()

        self.k = k
        self.intermediaire = intermediaire

        self.layers = torch.nn.ModuleList([NonLinearBE(layers_shapes[0], layers_shapes[1], k, init="ones", activation_RS=activationRS),
                                          torch.nn.ReLU(),
                                          torch.nn.Dropout(0.1)])

        for i in range(1,len(layers_shapes)-2):
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
        preds = torch.stack([head(X[:,i]) for i, head in enumerate(self.pred_heads)], dim=1)
        
        if self.mean_over_heads:
            return preds.mean(dim=1)
        return preds


class MLP_k(nn.Module):
    def __init__(self, MLP, k=32, mean_over_heads = True):
        super().__init__()
        self.k = k
        self.MLPs = nn.ModuleList([deepcopy(MLP) for _ in range(k)])
        self.mean_over_heads = mean_over_heads
    
    def forward(self, x):
        preds = torch.stack([MLP(x) for MLP in self.MLPs], dim=1)
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
        self.layers = nn.ModuleList([LinearBE(layers_shapes[0], layers_shapes[1], k, init="ones"),
                                     nn.ReLU(),
                                     nn.Dropout(0.1)])

        for i in range(1, len(layers_shapes) - 2):
            self.layers.append(LinearBE(layers_shapes[i], layers_shapes[i + 1], k, init, amplitude))
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
