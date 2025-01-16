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
    def __init__(self, input_dim, hidden_dim, output_dim, k):
        super(TabM, self).__init__()

        self.k = k
        self.shared_layer = nn.Linear(input_dim, hidden_dim) # Matrice W [input_dim, hidden_dim]

        self.adapters_r = nn.Parameter(init_adapters(k, hidden_dim).float())  # ri [k, hidden_dim]
        self.adapters_s = nn.Parameter(init_adapters(k, input_dim).float())   # si [k, input_dim]
        self.bias = nn.Parameter(torch.zeros(k, input_dim))  # B [k, input_dim]

        self.output_heads = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(k)])  # k tetes de sortie (une pour chaque adaptateur)
        self.dropout = nn.Dropout(0.2)  

    def forward(self, x):
        batch_size = x.size(0)  # [batch_size, input_dim]
        
        x_shared = torch.relu(self.shared_layer(x))  # [batch_size, hidden_dim]

        # Étendre les dimensions pour créer k copies de chaque entrée
        x_expanded = x_shared.unsqueeze(1).expand(batch_size, self.k, -1)  # [batch_size, k, hidden_dim]

        # Ajustez les dimensions des adaptateurs r et s pour traiter les données par batch
        R = self.adapters_r.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, k, hidden_dim]
        S = self.adapters_s.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, k, input_dim]
        B = self.bias.unsqueeze(0).expand(batch_size, -1, -1)        # [batch_size, k, input_dim]

        # Calcul matriciel du BatchEnssemble
        # (x_expanded * R) est de forme [batch_size, k, hidden_dim], self.shared_layer.weight est de forme [hidden_dim, input_dim] --> Le produit matriciel donne [batch_size, k, input_dim]
        x_adapted = (x_expanded * R) @ self.shared_layer.weight  # [batch_size, k, input_dim]
        x_adapted = x_adapted * S + B  # [batch_size, k, input_dim]

        # Calcul des sorties pour chaque modèle
        outputs = [self.output_heads[i](x_adapted[:, i, :]) for i in range(self.k)]  # Liste de [batch_size, output_dim] (une sortie pour chacun des k sous modèle)
        
        return torch.stack(outputs, dim=1)  # [batch_size, k, output_dim]


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


# ----- Training ------#

def train_model(net, train_loader, test_loader, log_dir, criterion=RMSELoss(), lr=1e-3, nb_iter=100, grad_clip=1.0):
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr )

    writer = SummaryWriter(log_dir=log_dir)
    
    for epoch in tqdm(range(nb_iter)):
        net.train()
        train_losses = []
        for x, y in train_loader:
            # x, y = x.to(device), y.to(device)
            preds = net(x)
            preds = preds.mean(dim=1) if isinstance(net, TabM) else preds
            loss = criterion(preds, y)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)  # Gradient clipping
            optimizer.step()

        writer.add_scalar("Loss/train", np.mean(train_losses), epoch)

        net.eval()
        test_losses = []
        with torch.no_grad():
            for x, y in test_loader:
                preds = net(x)
                preds = preds.mean(dim=1) if isinstance(net, TabM) else preds
                loss = criterion(preds, y)
                test_losses.append(loss.item())

        writer.add_scalar("Loss/test", np.mean(test_losses), epoch)



if __name__ == "__main__":
    # Dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    BATCH_SIZE = 256
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialisations
    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = 1
    k = 4

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tabM = TabM(input_dim, hidden_dim, output_dim, k)
    simple_mlp = SimpleMLP(input_dim, hidden_dim, output_dim)

    # Train et Eval des modèles
    train_model(tabM, train_loader, test_loader, "runs/TabM_California")
    train_model(simple_mlp, train_loader, test_loader, "runs/MLP_California")

