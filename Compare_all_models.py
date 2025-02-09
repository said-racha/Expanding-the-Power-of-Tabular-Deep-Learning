import time
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_auc_score, mean_squared_error, r2_score
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_iris, fetch_openml
import xgboost as xgb
import lightgbm as lgb
from ucimlrepo import fetch_ucirepo


# import excel_former
# import ft_transformer
# import t2g_former


#***************************************************** MODELS ***************************************************************#
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
    def __init__(self, in_features: int, hidden_sizes: int, k=32, dropout_rate=0.1):
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
    def __init__(self, in_features: int, hidden_sizes: int, k=32, dropout_rate=0.1):
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
    def __init__(self, in_features: int, hidden_sizes: int, k=32, dropout_rate=0.1):
        super().__init__()

        self.k = k

        layer_sizes = [in_features] + hidden_sizes

        layers = [linear_BE(layer_sizes[i], layer_sizes[i+1], k, dropout_rate, initialize_to_1=True) for i in range(len(layer_sizes)-1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        return self.layers(X)


class MLPk(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: int, k=32, dropout_rate=0.1):
        super().__init__()

        layer_sizes = [in_features] + hidden_sizes

        layers = [MLPk_layer(layer_sizes[i], layer_sizes[i+1], k, dropout_rate) for i in range(len(layer_sizes)-1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        return self.layers(X)

# ===== MODELS =====


class MLP(nn.Module):
    """
    Simple MLP model
    """

    def __init__(self, in_features: int, hidden_sizes: int, out_features: int, dropout_rate=0.1):
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

    def __init__(self, backbone: nn.Module, in_features: int, hidden_sizes: int, out_features: int, k=32, dropout_rate=0.1, mean_over_heads=True):
        super().__init__()

        self.backbone = backbone(in_features, hidden_sizes, k, dropout_rate)
        self.in_features = in_features
        self.k = k

        self.mean_over_heads = mean_over_heads

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

        if self.mean_over_heads:
            preds = preds.mean(dim=1)

        return preds


#============= ATTENTION MODEL ===============#
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
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

# Define the TabM model with Attention
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
            *[linear_BE(layer_sizes[i], layer_sizes[i + 1], k, dropout_rate, initialize_to_1=True) for i in range(len(layer_sizes) - 1)]
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

#***************************************************** DATASETS ***************************************************************#

def get_breast_cancer_data(split=0.2, batch_size=32, seed=42):
    """
    Loads the Breast Cancer dataset into dataloaders.
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed, stratify=y)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()), batch_size=batch_size, shuffle=False)
    shape_x = X_train.shape[1]
    shape_y = len(set(y))
    return train_loader, test_loader, shape_x, shape_y

def get_adult_income_data(split=0.2, batch_size=32, seed=42):
    """
    Loads the Adult Income dataset into dataloaders.
    """
    data = pd.read_csv("adult.csv")
    X = data.drop(columns='income')
    y = data['income']
    X = pd.get_dummies(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed, stratify=y)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()), batch_size=batch_size, shuffle=False)
    shape_x = X_train.shape[1]
    shape_y = len(set(y))

    return train_loader, test_loader, shape_x, shape_y

def get_california_housing_data(split=0.2, batch_size=32, seed=42):
    """
    Loads the California Housing dataset into dataloaders.
    """
    data = fetch_california_housing()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float()), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float()), batch_size=batch_size, shuffle=False)
    shape_x = X_train.shape[1]
    shape_y = 1  # Regression problem
    return train_loader, test_loader, shape_x, shape_y

def get_iris_data(split=0.2, batch_size=32, seed=42):
    """
    Loads the Iris dataset into dataloaders.
    """
    data = load_iris()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed, stratify=y)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()), batch_size=batch_size, shuffle=False)
    shape_x = X_train.shape[1]
    shape_y = len(set(y))
    return train_loader, test_loader, shape_x, shape_y

def get_quality_wine_data(split=.2, batch_size=32, seed=42):
    """
    Loads the Wine Quality dataset into dataloaders.
    """

    wine_quality = fetch_ucirepo(id=186)

    X = wine_quality.data.features
    y = wine_quality.data.targets

    # Normalize
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed, stratify=y)

    # Remap the labels to be in the range [0, num_classes - 1]
    unique_labels = np.unique(y)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}

    # Convert y_train and y_test to Series if they are DataFrames
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze() 
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze() 

    y_train = y_train.map(label_mapping)
    y_test = y_test.map(label_mapping)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train.to_numpy()).float(), torch.tensor(y_train.to_numpy()).long().squeeze()),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_test.to_numpy()).float(), torch.tensor(y_test.to_numpy()).long().squeeze()),
        batch_size=batch_size, shuffle=False
    )

    shape_x = X_train.shape[1] 
    shape_y = len(np.unique(y))  # Number of classes

    return train_loader, test_loader, shape_x, shape_y



#***************************************************** TRAINING ***************************************************************#


def train_model(net, train_loader, test_loader, log_dir, device, task_type="binary", criterion=None, lr=1e-4, nb_iter=15):
    """
    Trains a flexible PyTorch model for regression, binary or multiclass classification.

    Args:
        net: PyTorch model.
        train_loader: DataLoader for training.
        test_loader: DataLoader for testing.
        log_dir: Directory to save logs.
        device: Device (CPU or GPU) to run the model.
        task_type: Type of task ("binary", "multiclass", "regression").
        criterion: Loss function (optional, defaults based on the task).
        lr: Learning rate.
        nb_iter: Number of iterations (epochs).

    Returns:
        A dictionary containing the accuracy, loss, F1-score, precision, recall, AUC, RMSE, R², and execution time.
    """
    start_time = time.time()

    if task_type == "binary":
        criterion = criterion or nn.BCEWithLogitsLoss()
    elif task_type == "multiclass":
        criterion = criterion or nn.CrossEntropyLoss()
    elif task_type == "regression":
        criterion = criterion or nn.MSELoss()
    else:
        raise ValueError("Invalid task_type. Choose from 'binary', 'multiclass', or 'regression'.")

    optim = torch.optim.AdamW(net.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in tqdm(range(nb_iter), desc="Training", unit="ep"):
        net.train()
        losses = []
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            if task_type == "binary":
                y = y.float().view(-1, 1)  # Ensure shape (batch_size, 1) for BCEWithLogitsLoss

            yhat = net(x)

            # Loss calculation
            loss = criterion(yhat, y)
            losses.append(loss.item())

            # Accuracy and F1-score only for classification
            if task_type == "binary":
                preds_classes = (torch.sigmoid(yhat) > 0.5).float()
                correct += (preds_classes == y).sum().item()
                total += y.size(0)
                all_preds.extend(preds_classes.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
            elif task_type == "multiclass":
                preds_classes = torch.argmax(yhat, dim=1)
                correct += (preds_classes == y).sum().item()
                total += y.size(0)
                all_preds.extend(preds_classes.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optim.step()

        writer.add_scalar("Loss/train", sum(losses) / len(losses), epoch)
        if task_type != "regression":
            writer.add_scalar("Accuracy/train", correct / total, epoch)
            f1 = f1_score(all_labels, all_preds, average='macro' if task_type == "multiclass" else 'binary')
            writer.add_scalar("F1/train", f1, epoch)

        # Evaluation
        net.eval()
        with torch.no_grad():
            test_losses = []
            test_correct = 0
            test_total = 0
            test_all_preds = []
            test_all_labels = []

            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                if task_type == "binary":
                    y = y.float().view(-1, 1)  # Ensure shape for BCEWithLogitsLoss

                yhat = net(x)

                loss = criterion(yhat, y)
                test_losses.append(loss.item())

                if task_type == "binary":
                    preds_classes = (torch.sigmoid(yhat) > 0.5).float()
                    test_correct += (preds_classes == y).sum().item()
                    test_total += y.size(0)
                    test_all_preds.extend(preds_classes.cpu().numpy())
                    test_all_labels.extend(y.cpu().numpy())
                elif task_type == "multiclass":
                    preds_classes = torch.argmax(yhat, dim=1)
                    test_correct += (preds_classes == y).sum().item()
                    test_total += y.size(0)
                    test_all_preds.extend(preds_classes.cpu().numpy())
                    test_all_labels.extend(y.cpu().numpy())
                elif task_type == "regression":
                  test_all_preds.extend(yhat.cpu().numpy())
                  test_all_labels.extend(y.cpu().numpy())


            writer.add_scalar("Loss/test", sum(test_losses) / len(test_losses), epoch)
            if task_type != "regression":
                writer.add_scalar("Accuracy/test", test_correct / test_total, epoch)
                f1 = f1_score(test_all_labels, test_all_preds, average='macro' if task_type == "multiclass" else 'binary')
                writer.add_scalar("F1/test", f1, epoch)

    writer.close()
    execution_time = time.time() - start_time

    # Calculate additional metrics
    if task_type != "regression":
        precision = precision_score(test_all_labels, test_all_preds, average='macro' if task_type == "multiclass" else 'binary')
        recall = recall_score(test_all_labels, test_all_preds, average='macro' if task_type == "multiclass" else 'binary')
        auc = roc_auc_score(test_all_labels, test_all_preds) if task_type == "binary" else None
        rmse = None
        r2 = None
    else:
        precision = None
        recall = None
        auc = None
        rmse = np.sqrt(mean_squared_error(test_all_labels, test_all_preds))
        mse = mean_squared_error(test_all_labels, test_all_preds) 
        r2 = r2_score(test_all_labels, test_all_preds)

    results = {
        "accuracy": test_correct / test_total if task_type != "regression" else None,
        "loss": sum(test_losses) / len(test_losses),
        "f1_score": f1_score(test_all_labels, test_all_preds, average='macro' if task_type == "multiclass" else 'binary') if task_type != "regression" else None,
        "precision": precision,
        "recall": recall,
        "auc_score": auc,
        "rmse": rmse,
        "r2": r2,
        "execution_time": execution_time
    }

    return results


def train_xgboost(X_train, y_train, X_test, y_test, task_type="binary"):
    """
    Trains an XGBoost model and returns evaluation metrics.
    """
    start_time = time.time()

    if task_type == "binary":
        model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    elif task_type == "multiclass":
        model = xgb.XGBClassifier(objective="multi:softmax", random_state=42)
    else:
        model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if task_type == "binary" else None

    execution_time = time.time() - start_time

    # Calculate metrics
    if task_type != "regression":
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro' if task_type == "multiclass" else 'binary')
        precision = precision_score(y_test, y_pred, average='macro' if task_type == "multiclass" else 'binary')
        recall = recall_score(y_test, y_pred, average='macro' if task_type == "multiclass" else 'binary')
        auc = roc_auc_score(y_test, y_proba) if task_type == "binary" else None
        rmse = None
        r2 = None
    else:
        accuracy = None
        f1 = None
        precision = None
        recall = None
        auc = None
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

    results = {
        "accuracy": accuracy,
        "loss": None,  # XGBoost does not provide a direct loss value
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "auc_score": auc,
        "rmse": rmse,
        "r2": r2,
        "execution_time": execution_time
    }

    return results

import warnings
warnings.filterwarnings('ignore')

def train_lightgbm(X_train, y_train, X_test, y_test, task_type="binary"):
    """
    Trains a LightGBM model and returns evaluation metrics.
    """
    start_time = time.time()

    if task_type == "binary":
        model = lgb.LGBMClassifier(random_state=42, verbosity=-1)
    elif task_type == "multiclass":
        model = lgb.LGBMClassifier(random_state=42, verbosity=-1)
    else:
        model = lgb.LGBMRegressor(random_state=42, verbosity=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if task_type == "binary" else None

    execution_time = time.time() - start_time

    # Calculate metrics
    if task_type != "regression":
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro' if task_type == "multiclass" else 'binary')
        precision = precision_score(y_test, y_pred, average='macro' if task_type == "multiclass" else 'binary')
        recall = recall_score(y_test, y_pred, average='macro' if task_type == "multiclass" else 'binary')
        auc = roc_auc_score(y_test, y_proba) if task_type == "binary" else None
        rmse = None
        r2 = None
    else:
        accuracy = None
        f1 = None
        precision = None
        recall = None
        auc = None
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

    results = {
        "accuracy": accuracy,
        "loss": None,  # LightGBM does not provide a direct loss value
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "auc_score": auc,
        "rmse": rmse,
        "r2": r2,
        "execution_time": execution_time
    }

    return results


#========== Main function ==========#

def initialize_device():
    """
    Initialize the device (CPU, GPU, or MPS)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device

def dataloader_to_numpy(dataloader):
    """
    Convert a PyTorch DataLoader to numpy arrays.
    """
    X_list, y_list = [], []

    for X, y in dataloader:
        X_list.append(X.numpy())
        y_list.append(y.numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y

# Run the main function
if __name__ == "__main__":
    BATCH_SIZE = 32
    device = initialize_device()

    # Load datasets and perform pruning and training for each
    datasets = [
        ("breast_cancer", get_breast_cancer_data, "binary", 1e-4, 25),
        ("adult_income", get_adult_income_data, "binary", 1e-4, 10),
        ("california_housing", get_california_housing_data, "regression", 1e-4, 15),
        ("iris", get_iris_data, "multiclass", 1e-4, 50), 
        ("wine_quality", get_quality_wine_data, "multiclass", 1e-4, 20) 
    ]

    results = []

    for dataset_name, loader_func, task_type, lr, nb_epochs in datasets:

        print(f"\n\n *** Processing {dataset_name} dataset...")
        train_loader, test_loader, shape_x, shape_y = loader_func(split=0.2, batch_size=BATCH_SIZE, seed=42)

        # Convert DataLoader to numpy arrays for XGBoost and LightGBM
        X_train, y_train = dataloader_to_numpy(train_loader)
        X_test, y_test = dataloader_to_numpy(test_loader)

        # Set hidden layers and criterion based on task type (multiclass, binary, regression)
        if task_type == "multiclass":
            layers = [64, 32, 16]
            criterion = nn.CrossEntropyLoss()
            out_features = shape_y
        elif task_type == "binary":
            layers = [64, 32]
            criterion = nn.BCEWithLogitsLoss()
            out_features = 1
        else:
            layers = [64, 32]
            criterion = nn.MSELoss()
            out_features = 1

        # Compare models
        input_dim = shape_x
        hidden_sizes = layers
        output_dim = out_features

        tabM_naive = EnsembleModel(TabM_naive, input_dim, hidden_sizes, output_dim, dropout_rate=0).to(device)
        simple_MLP = MLP(input_dim, hidden_sizes, output_dim, dropout_rate=0).to(device)
        tabM_mini = EnsembleModel(TabM_mini, input_dim, hidden_sizes, output_dim, dropout_rate=0).to(device)
        tabM = EnsembleModel(TabM, input_dim, hidden_sizes, output_dim, dropout_rate=0).to(device)
        mlpk = EnsembleModel(MLPk, input_dim, hidden_sizes, output_dim, dropout_rate=0).to(device)

        embed_dim = 12
        tabM_attention = TabMWithAttention(input_dim, hidden_sizes, embed_dim, output_dim=output_dim, num_heads=2, k=32, dropout_rate=0.3).to(device)

        # # Define baselines (ExcelFormer, FT-Transformer, T2g-Former)
        # model_config = {
        #     "d_numerical": input_dim,
        #     "categories": [],
        #     "token_bias": True,
        #     "n_layers": 3,
        #     "d_token": 256,
        #     "n_heads": 32,
        #     "attention_dropout": 0.3,
        #     "ffn_dropout": 0.0,
        #     "residual_dropout": 0.0,
        #     "prenormalization": True,
        #     "kv_compression": None,
        #     "kv_compression_sharing": None,
        #     "d_out": output_dim,
        # }

        # ft = ft_transformer.FTTransformer(**model_config, d_ffn_factor=.6, activation="gelu", initialization="xavier").to(device)
        # excel = excel_former.ExcelFormer(**model_config).to(device)
        # t2g = t2g_former.T2GFormer(**model_config, d_ffn_factor=.6, activation="gelu", initialization="xavier").to(device)

        NB_ITER = nb_epochs

        # Training
        models = {
            # "TabM_naive": tabM_naive,
            # "Simple_MLP": simple_MLP,
            # "TabM_mini": tabM_mini,
            "TabM": tabM,
            "MLPk": mlpk,
            "TabM_Attention": tabM_attention,
            # "FT-Transformer": ft,
            # "Excel-Former": excel,
            # "T2g-Former": t2g
        }

        for model_name, model in models.items():
            print(f"Training {model_name}...")
            result = train_model(model, train_loader, test_loader, f'runs/{dataset_name}/{model_name}', device, task_type=task_type, nb_iter=NB_ITER)
            results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Accuracy": result["accuracy"],
                "Loss": result["loss"],
                "F1-Score": result["f1_score"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "AUC Score": result["auc_score"],
                "RMSE": result["rmse"],
                "R²": result["r2"],
                "Execution Time": result["execution_time"]
            })

 
        print("Training XGBoost...")
        xgb_result = train_xgboost(X_train, y_train, X_test, y_test, task_type=task_type)
        results.append({
            "Dataset": dataset_name,
            "Model": "XGBoost",
            "Accuracy": xgb_result["accuracy"],
            "Loss": xgb_result["loss"],
            "F1-Score": xgb_result["f1_score"],
            "Precision": xgb_result["precision"],
            "Recall": xgb_result["recall"],
            "AUC Score": xgb_result["auc_score"],
            "RMSE": xgb_result["rmse"],
            "R²": xgb_result["r2"],
            "Execution Time": xgb_result["execution_time"]
        })

        print("Training LightGBM...")
        lgb_result = train_lightgbm(X_train, y_train, X_test, y_test, task_type=task_type)
        results.append({
            "Dataset": dataset_name,
            "Model": "LightGBM",
            "Accuracy": lgb_result["accuracy"],
            "Loss": lgb_result["loss"],
            "F1-Score": lgb_result["f1_score"],
            "Precision": lgb_result["precision"],
            "Recall": lgb_result["recall"],
            "AUC Score": lgb_result["auc_score"],
            "RMSE": lgb_result["rmse"],
            "R²": lgb_result["r2"],
            "Execution Time": lgb_result["execution_time"]
        })

    results_df = pd.DataFrame(results)
    display(results_df)
