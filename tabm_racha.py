import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def init_adapters(k, size):
    return torch.randn(k, size)

class TabM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k=32):
        super(TabM, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.k = k
        self.shared_layer = nn.Linear(input_dim, hidden_dim)  # Matrice W [input_dim, hidden_dim]

        self.adapters_r = nn.Parameter(init_adapters(k, hidden_dim).float())  # ri [k, hidden_dim]
        self.adapters_s = nn.Parameter(init_adapters(k, input_dim).float())  # si [k, input_dim]
        self.bias = nn.Parameter(init_adapters(k, input_dim))  # B [k, input_dim]

        self.output_heads = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(k)])  # k tetes de sortie
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.size(0)  # [batch_size, input_dim]
        
        x_shared = torch.relu(self.shared_layer(x))  # [batch_size, hidden_dim]

        # Étendre les dimensions pour créer k copies de chaque entrée
        x_expanded = x_shared.unsqueeze(1).expand(batch_size, self.k, -1)  # [batch_size, k, hidden_dim]

        # Ajustez les dimensions des adaptateurs r et s pour traiter les données par batch
        R = self.adapters_r.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, k, hidden_dim]
        S = self.adapters_s.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, k, input_dim]
        B = self.bias.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, k, input_dim]

        # Calcul matriciel du BatchEnssemble
        x_adapted = (x_expanded * R) @ self.shared_layer.weight  # [batch_size, k, input_dim]
        x_adapted = x_adapted * S + B  # [batch_size, k, input_dim]

        # Calcul des sorties pour chaque modèle (tête)
        outputs = [self.output_heads[i](x_adapted[:, i, :]) for i in range(self.k)]  # [batch_size, k, output_dim]

        # Moyenne sur la dimension k
        # Pour que chaque sortie soit un vecteur de taille [batch_size, output_dim] représentant les scores pour chaque classe
        output = torch.mean(torch.stack(outputs, dim=1), dim=1)  # [batch_size, output_dim]

        return output



class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim)) 
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
