
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.datasets import load_breast_cancer, fetch_openml, fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import copy
import matplotlib.pyplot as plt


#*********************************************************** MODELS ***********************************************************#

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


#=========== PRUNING ===========#

class PrunableEnsembleModel(nn.Module):
    """
    Ensemble model with pruning mechanism:
    - Measures the individual contribution of each sub-model.
    - Allows for progressively removing the least performing sub-models.
    """
    def __init__(self, backbone: nn.Module, in_features: int, hidden_sizes: int, out_features: int, k=32, dropout_rate=0.1):
        super().__init__()
        self.backbone = backbone(in_features, hidden_sizes, k, dropout_rate)
        self.in_features = in_features
        self.hidden_sizes = hidden_sizes
        self.k = k
        self.out_features = out_features

        # Sub-model prediction (does not average like in the EnsembleModel class)
        self.pred_heads = nn.ModuleList([nn.Linear(hidden_sizes[-1], out_features) for _ in range(k)])

    def forward(self, X: torch.Tensor):
        """
        Forward pass:
        - The predictions of each sub-model are concatenated.
        """
        # Replicate X for each sub-model
        X = X.unsqueeze(1).repeat(1, self.k, 1)

        # Pass through the backbone
        X = self.backbone(X)  # (batch, k, hidden_sizes[-1])

        # Predictions from each sub-model
        preds = [head(X[:, i]) for i, head in enumerate(self.pred_heads)]
        preds = torch.stack(preds, dim=1)  # (batch, k, out_features)

        return preds

    def prune(self, X: torch.Tensor, y: torch.Tensor, keep_ratio=0.5, task_type="binary", verbose=False):
        """
        Pruning the sub-models:
        - Keeps a given ratio of the best performing sub-models.
        - Removes less performing sub-models.
        - Adapts to binary classification, multiclass, or regression tasks.

        Args:
            X : Input data.
            y : Labels or target values.
            keep_ratio : Ratio of sub-models to keep.
            task_type : Task type ('binary', 'multiclass', or 'regression').
        """
        if not (0 < keep_ratio <= 1):
            raise ValueError("keep_ratio must be a float between 0 and 1.")

        with torch.no_grad():

            if task_type == "binary":
                criterion = nn.BCEWithLogitsLoss(reduction="none")
                y = y.float().view(-1, 1)
            elif task_type == "multiclass":
                criterion = nn.CrossEntropyLoss(reduction="none")
            elif task_type == "regression":
                criterion = nn.MSELoss(reduction="none")
            else:
                raise ValueError("task_type must be one of 'binary', 'multiclass', or 'regression'.")

            # Loss
            losses = []
            preds = self.forward(X)  # (batch, k, out_features)
            if self.k > 1 :
              for i in range(self.k):
                  if task_type == "multiclass":
                      loss = criterion(preds[:, i, :], y)
                  else:
                      loss = criterion(preds[:, i, :].reshape(-1, 1), y.reshape(-1, 1))
                  losses.append(loss.mean().item())  # Mean loss for each sub-model

              # Sort the sub-models by loss (increasing loss)
              sorted_indices = sorted(range(self.k), key=lambda i: losses[i])
              keep_count = max(1, int(self.k * keep_ratio))
              keep_indices = sorted_indices[:keep_count]

              # Update the sub-models and their parameters
              self.pred_heads = nn.ModuleList([self.pred_heads[i] for i in keep_indices])
              self.k = keep_count

              # Prune the backbone parameters
              for layer in self.backbone.layers:
                  if hasattr(layer, "R") and hasattr(layer, "S") and hasattr(layer, "B"):
                      layer.R = nn.Parameter(layer.R[keep_indices])
                      layer.S = nn.Parameter(layer.S[keep_indices])
                      layer.B = nn.Parameter(layer.B[keep_indices])
              if verbose :
                print(f"Pruning performed: {len(sorted_indices) - keep_count} sub-models removed. {self.k} remaining.")
            else :
              if verbose :
                print(f"Only {self.k} sub-model remaining.")


#*********************************************************** TRANING (with pruning) ***********************************************************#

# Function to prune the model
def test_pruning(model, train_loader, keep_ratios, task_type="binary", verbose=False):
    """
    Test the pruning mechanism.
    - Gradually reduce the sub-models.
    - Measure overall accuracy after each step.

    Args:
        model (nn.Module): The prunable model.
        train_loader (DataLoader): DataLoader for test data.
        keep_ratios (list): List of model retention ratios.
        task_type (str): Task type ('binary', 'multiclass', or 'regression').

    Returns:
        None
    """
    for keep_ratio in keep_ratios:
        print(f"\nPruning with keep_ratio = {keep_ratio}")

        # Prune the model (using one batch to select sub-models)
        for X_batch, y_batch in train_loader:
            model.prune(X_batch, y_batch, keep_ratio, task_type=task_type, verbose=verbose)
            break

        # Evaluate performance after pruning
        total_correct, total_samples, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                preds = model.forward(X_batch)

                if task_type == "binary":
                    preds = preds.mean(dim=1)
                    preds_binary = torch.sigmoid(preds)
                    preds_binary = (preds_binary > 0.5)
                    total_correct += (preds_binary == y_batch.reshape(-1, 1)).sum().item()

                elif task_type == "multiclass":
                    preds = preds.mean(dim=1)
                    preds_class = preds.argmax(dim=1)
                    total_correct += (preds_class == y_batch).sum().item()

                elif task_type == "regression":
                    preds = preds.mean(dim=1)
                    total_loss += nn.functional.mse_loss(preds.reshape(-1,1), y_batch.reshape(-1,1)).item()

                total_samples += y_batch.size(0)

        if task_type in ["binary", "multiclass"]:
            accuracy = total_correct / total_samples
            print(f"Overall accuracy after pruning: {accuracy:.4f}")
        elif task_type == "regression":
            avg_loss = total_loss / total_samples
            print(f"Average loss after pruning: {avg_loss:.4f}")

# Training function with pruning

def train_with_pruning(
    model, train_loader, test_loader, optimizer, criterion,
    epochs=10, keep_ratios=[1.0, 0.75, 0.5, 0.25], filepath="best_model.pth",
    dataset_name="", task_type="binary", verbose=False
):
    best_loss = float('inf')
    best_model = None
    best_config = {}

    # Lists to store metrics for plotting
    train_losses = []
    test_losses = []
    accuracies = []

    
    for epoch in range(epochs):
        model.train()
        epoch_loss, total_samples = 0.0, 0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(X_batch)

            if task_type == 'binary':
                preds = outputs.mean(dim=1)
                loss = criterion(preds, y_batch.unsqueeze(1).float())

            elif task_type == 'multiclass':
                preds = outputs.mean(dim=1)
                loss = criterion(preds, y_batch)

            elif task_type == 'regression':
                preds = outputs.mean(dim=1)
                loss = criterion(preds.reshape(-1,1), y_batch.reshape(-1,1).float())

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_samples += y_batch.size(0)

        avg_epoch_loss = epoch_loss / total_samples
        print(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss}")

        
        # Test pruning at different keep ratios
        for keep_ratio in keep_ratios:
            model_copy = copy.deepcopy(model)
            model_copy.prune(X_batch, y_batch, keep_ratio=keep_ratio, task_type=task_type, verbose=verbose)

            model_copy.eval()
            total_loss, total_correct, total_samples = 0.0, 0, 0

            with torch.no_grad():
                for X_batch, y_batch in train_loader:
                    outputs = model_copy(X_batch)

                    if task_type == "binary":
                        preds = outputs.mean(dim=1)
                        loss = criterion(preds, y_batch.unsqueeze(1).float())
                        probs = torch.sigmoid(preds)
                        predictions = (probs >= 0.5).float()
                        total_correct += (predictions == y_batch.unsqueeze(1)).float().sum()

                    elif task_type == "multiclass":
                        preds = outputs.mean(dim=1)
                        preds_class = preds.argmax(dim=1)
                        loss = criterion(preds, y_batch)
                        total_correct += (preds_class == y_batch).sum().item()

                    elif task_type == "regression":
                        preds = outputs.mean(dim=1)
                        loss += criterion(preds.reshape(-1,1), y_batch.reshape(-1,1).float())

                    total_loss += loss.item()
                    total_samples += y_batch.size(0)

            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples

            
            if task_type in ["binary", "multiclass"]:
                print(f"Pruning with keep_ratio={keep_ratio}, Accuracy: {accuracy:.4f}")
            elif task_type == "regression":
                print(f"Pruning with keep_ratio={keep_ratio}, Train Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = model_copy
                best_config = {"k": model_copy.k}

            if (model.k == 1):  # Test the pruning only if there is at least 2 sub-models
                break

        model = best_model
        train_losses.append(best_loss)

        # Evaluate on test set after each epoch
        model.eval()
        test_loss, test_correct, test_samples = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)

                if task_type == "binary":
                    preds = outputs.mean(dim=1)
                    loss = criterion(preds, y_batch.unsqueeze(1).float())
                    probs = torch.sigmoid(preds)
                    predictions = (probs >= 0.5).float()
                    test_correct += (predictions == y_batch.unsqueeze(1)).float().sum()

                elif task_type == "multiclass":
                    preds = outputs.mean(dim=1)
                    preds_class = preds.argmax(dim=1)
                    loss = criterion(preds, y_batch)
                    test_correct += (preds_class == y_batch).sum().item()

                elif task_type == "regression":
                    preds = outputs.mean(dim=1)
                    loss += criterion(preds.reshape(-1,1), y_batch.reshape(-1,1).float())

                test_loss += loss.item()
                test_samples += y_batch.size(0)

        avg_test_loss = test_loss / test_samples
        test_losses.append(avg_test_loss)
        accuracy = test_correct / test_samples
        accuracies.append(accuracy)

        if verbose:
            if task_type == "binary" or task_type == "multiclass":
                print(f"Epoch {epoch + 1}, Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}")
            elif task_type == "regression":
                print(f"Epoch {epoch + 1}, Test Loss: {avg_test_loss:.4f}")

    # Plot the training and test loss curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()

    if task_type in ["binary", "multiclass"]:
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), accuracies, label="Accuracy", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Test Accuracy")
        plt.legend()

    plt.tight_layout()
    plt.show()

    if verbose:
        print(f"Best configuration of the model {best_config}")

    return model, best_config

# Save /load the model function
def save_model(model, filepath, config):
    """
    Save the PyTorch model and its configuration.

    Args:
        model (nn.Module): The model to save.
        filepath (str): Path to save the model.
        config (dict): Configuration (e.g., k after pruning).

    Returns:
        None
    """
    config.update({
        "backbone": model.backbone.__class__,
        "in_features": model.in_features,
        "hidden_sizes": model.hidden_sizes,
        "out_features": model.out_features,
        "k": model.k
    })

    torch.save({
        "state_dict": model.state_dict(),
        "config": config
    }, filepath)
    print(f"Model and configuration saved to {filepath}")

def load_model(filepath, model_class, default_config):
    """
    Load a PyTorch model from a file.

    Args:
        filepath (str): Path to the saved model file.
        model_class (nn.Module): The class of the model to load.
        default_config (dict): Default configuration of the model.

    Returns:
        nn.Module: The loaded model, ready to be used.
    """
    # Load the file
    checkpoint = torch.load(filepath)  
    state_dict = checkpoint["state_dict"]
    config = checkpoint.get("config", default_config)  # Use the saved configuration or the default one

    # Reinitialize the model with the correct configuration
    model = model_class(**config)

    # Load the model parameters
    model.load_state_dict(state_dict)

    model.eval()

    print(f"Model loaded from {filepath}")
    return model

#========== Dataset functions ==========#
def get_breast_cancer_data(split=0.2, batch_size=32, seed=42):
    """
    Loads the Breast Cancer dataset into dataloaders.
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()), batch_size=batch_size, shuffle=False)
    shape_x = X_train.shape[1]
    shape_y = len(set(y))
    return train_loader, test_loader, shape_x, shape_y


#========== Main function ==========#
def main():
    BATCH_SIZE = 32

    # Load datasets and perform pruning and training for each
    datasets = [
        ("breast_cancer", get_breast_cancer_data, "binary", 1e-4, 8),
        ("adult_income", get_adult_income_data, "binary", 1e-4, 15),
        ("california_housing", get_california_housing_data, "regression", 1e-4, 8),
        ("iris", get_iris_data, "multiclass", 1e-4, 12)
    ]

    for dataset_name, loader_func, task_type, lr, nb_epochs in datasets:
        print(f"\n\n *** Processing {dataset_name} dataset...")
        train_loader, test_loader, shape_x, shape_y = loader_func(split=0.2, batch_size=BATCH_SIZE, seed=42)

        # Set hidden layers based on task type (multiclass, binary, regression)
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

        
        # Initialize the model
        prunable_model = PrunableEnsembleModel(TabM, in_features=shape_x, hidden_sizes=layers, out_features=out_features, k=32)

        # Test pruning (adjust this for task_type)
        keep_ratios = [1, 0.75, 0.5, 0.25, 0.1]
        test_pruning(prunable_model, train_loader, keep_ratios, task_type, verbose=True)

        # Train the model with pruning
        prunable_model = PrunableEnsembleModel(TabM, in_features=shape_x, hidden_sizes=layers, out_features=out_features, k=32)

        optimizer = optim.Adam(prunable_model.parameters(), lr=lr)

        trained_model, best_config = train_with_pruning(
            prunable_model, train_loader, test_loader, optimizer, criterion, epochs=nb_epochs, task_type=task_type, verbose=True
        )

        # Save the best model
        model_name = f"{dataset_name}_best_model.pth"
        save_model(trained_model, model_name, best_config)

        
        # Test loading the model
        default_config = {
            "backbone": TabM,
            "in_features": shape_x,
            "hidden_sizes": layers,
            "out_features": out_features,
            "dropout_rate": 0.1
        }

        default_config.update(best_config)

        loaded_model = load_model(model_name, PrunableEnsembleModel, default_config)

        print("Loading the model ...")
      
        loaded_model.eval()
        test_loss, test_correct, test_samples = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = loaded_model(X_batch)

                if task_type == "binary":
                    preds = outputs.mean(dim=1)
                    loss = criterion(preds, y_batch.unsqueeze(1).float())
                    probs = torch.sigmoid(preds)
                    predictions = (probs >= 0.5).float()
                    test_correct += (predictions == y_batch.unsqueeze(1)).float().sum()

                elif task_type == "multiclass":
                    preds = outputs.mean(dim=1)
                    preds_class = preds.argmax(dim=1)
                    loss = criterion(preds, y_batch)
                    test_correct += (preds_class == y_batch).sum().item()

                elif task_type == "regression":
                    preds = outputs.mean(dim=1)
                    loss += criterion(preds.reshape(-1,1), y_batch.reshape(-1,1).float())

                test_loss += loss.item()
                test_samples += y_batch.size(0)

        avg_test_loss = test_loss / test_samples
        accuracy = test_correct / test_samples
        
        if task_type == "binary" or task_type == "multiclass":
            print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}")
        elif task_type == "regression":
            print(f"Test Loss: {avg_test_loss:.4f}")

if __name__ == "__main__":
    main()

