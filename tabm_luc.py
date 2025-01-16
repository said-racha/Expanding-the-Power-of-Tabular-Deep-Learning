
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ===== LAYERS =====

class linear_BE(nn.Module):
    def __init__(self, in_features: int, out_features: int, k=32, dropout_rate=0.1, initialize_to_1=False):
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
    def __init__(self, in_features: int, out_features: int, dropout_rate=0.1):
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
    def __init__(self, in_features: int, out_features: int, k=32, dropout_rate=0.1):
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


# ===== BACKBONES =====


class TabM_naive(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: int, k=32):
        super().__init__()

        self.in_features = in_features
        self.hidden_sizes = hidden_sizes
        self.k = k

        layer_sizes = [in_features] + hidden_sizes

        layers = [linear_BE(layer_sizes[i], layer_sizes[i+1], k) for i in range(len(layer_sizes)-1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        return self.layers(X)


class TabM_mini(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: int, k=32):
        super().__init__()

        self.k = k

        self.R = nn.Parameter(torch.zeros((k, in_features)))
        nn.init.uniform_(self.R, -1, 1)

        layer_sizes = [in_features] + hidden_sizes

        layers = [MLP_layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        output = X * self.R
        return self.layers(output)


class TabM(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: int, k=32):
        super().__init__()

        self.k = k

        layer_sizes = [in_features] + hidden_sizes

        layers = [linear_BE(layer_sizes[i], layer_sizes[i+1], k, initialize_to_1=True) for i in range(len(layer_sizes)-1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        return self.layers(X)


class MLPk(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: int, k=32):
        super().__init__()

        layer_sizes = [in_features] + hidden_sizes

        layers = [MLPk_layer(layer_sizes[i], layer_sizes[i+1], k) for i in range(len(layer_sizes)-1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        return self.layers(X)

# ===== MODELS =====


class MLP(nn.Module):
    """
    Simple MLP model
    """

    def __init__(self, in_features: int, hidden_sizes: int, out_features: int):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features

        layer_sizes = [in_features] + hidden_sizes + [out_features]

        layers = [*[MLP_layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)],
                  nn.Linear(layer_sizes[-1], out_features)
                  ]

        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        return self.layers(X).reshape(-1) # Temp


class EnsembleModel(nn.Module):
    """
    Global ensemble model that : 
    - takes batched input (batch, in_features)
    - clones it k times (batch, k, in_features)
    - passes it through a backbone (which model you want e.g TabM, MLPk, etc.) (batch, k, hidden_sizes[-1])
    - passes the output through k prediction heads, mean over heads (batch, out_features)
    """

    def __init__(self, backbone: nn.Module, in_features: int, hidden_sizes: int, out_features: int, k=32):
        super().__init__()

        self.backbone = backbone(in_features, hidden_sizes, k)
        self.in_features = in_features
        self.k = k

        self.pred_heads = nn.ModuleList([nn.Linear(hidden_sizes[-1], out_features) for _ in range(k)])

    def forward(self, X: torch.Tensor):
        # clone X to shape (batch, k, dim)
        X = X.unsqueeze(1).repeat(1, self.k, 1)

        # pass through backbone
        X = self.backbone(X)

        # pass through prediction heads
        preds = [head(X[:, i]) for i, head in enumerate(self.pred_heads)]

        return torch.cat(preds, dim=1).mean(dim=1)


# ===== TEST =====

# Training

BATCH_SIZE = 25


def train_cancer(net, train_loader, test_loader, log_dir, criterion=nn.BCEWithLogitsLoss(), lr=1e-3, nb_iter=50):

    optim = torch.optim.Adam(net.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in tqdm(range(nb_iter)):

        net.train()
        losses = []
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            preds = net(x)
            loss = criterion(preds, y)
            losses.append(loss)

            # Compute accuracy
            preds_binary = (torch.sigmoid(preds) > 0.5).int()
            train_correct += (preds_binary == y).sum()
            train_total += y.size(0)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optim.step()

        writer.add_scalar("Loss/train", torch.tensor(losses).mean(), epoch)
        writer.add_scalar("Accuracy/train", train_correct / train_total, epoch)

        net.eval()
        with torch.no_grad():
            test_losses = []
            test_correct = 0
            test_total = 0

            for x, y in test_loader:
                preds = net(x)
                loss = criterion(preds, y)
                test_losses.append(loss)

                # Compute accuracy
                preds_binary = (torch.sigmoid(preds) > 0.5).int()
                test_correct += (preds_binary == y).sum()
                test_total += y.size(0)

            writer.add_scalar("Loss/test", torch.tensor(test_losses).mean(), epoch)
            writer.add_scalar("Accuracy/test", test_correct / test_total, epoch)


# Classification : `breast_cancer`
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=BATCH_SIZE, shuffle=False)


layers = [64, 32, 16]
tabM_naive = EnsembleModel(TabM_naive, X_train.shape[1], layers, 1)
simple_MLP = MLP(X_train.shape[1], layers, 1)
tabM_mini = EnsembleModel(TabM_mini, X_train.shape[1], layers, 1)
tabM = EnsembleModel(TabM, X_train.shape[1], layers, 1)
mlpk = EnsembleModel(MLPk, X_train.shape[1], layers, 1)


# print(tabM_naive)

train_cancer(tabM_naive, train_loader, test_loader, 'runs/TabM_naive')
train_cancer(simple_MLP, train_loader, test_loader, 'runs/MLP')
train_cancer(tabM_mini, train_loader, test_loader, 'runs/TabM_mini')
train_cancer(tabM, train_loader, test_loader, 'runs/TabM')
train_cancer(mlpk, train_loader, test_loader, 'runs/MLPk')
