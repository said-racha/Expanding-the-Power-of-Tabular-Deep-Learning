from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import optuna
import torch
import torch.nn as nn

from torchmetrics import MeanSquaredError
from torchmetrics import R2Score

import math

from torch.utils.tensorboard import SummaryWriter


from tabm_luc import *


import json

from tqdm import tqdm


def train_regression(mlp, train_loader, val_loader, device, optimizer=None, loss_fn=None, epochs=20, log_dir=None):
    mlp.to(device)

    if not optimizer:
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    if not loss_fn:
        loss_fn = nn.MSELoss().to(device)

    if log_dir:
        writer = SummaryWriter(log_dir=log_dir)

    for epoch in tqdm(range(epochs)):
        mlp.train()

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            y_hat = mlp(X)
            # y_hat = output[:, 0] + torch.randn_like(output[:, 0]) * torch.sqrt(output[:, 1])
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model
        mlp.eval()
        if log_dir:
            metrics_train = eval_scores_regression(mlp, train_loader, device)
            writer.add_scalar("RMSE/train", metrics_train['RMSE'], epoch)
            writer.add_scalar("R2/train", metrics_train['R2'], epoch)

            metrics_val = eval_scores_regression(mlp, val_loader, device)
            writer.add_scalar("RMSE/val", metrics_val['RMSE'], epoch)
            writer.add_scalar("R2/val", metrics_val['R2'], epoch)

    return mlp


def eval_regression(mlp, train_fn, train_loader, val_loader, test_loader, device, model_name):
    # Train the model with the best hyperparameters
    mlp = train_fn(mlp, train_loader, val_loader, model_name, device=device)

    # Evaluate the model
    metrics = eval_scores_regression(mlp, test_loader, device)

    with open(f"results/{model_name}.txt", "w") as f:
        json.dump(metrics, f)


def optimize_hyperparams_MLP(train_fn, train_loader, val_loader, in_features, out_features, device):
    # Define the objective function
    def objective(trial):
        # Define the model
        mlp = MLP(in_features=in_features, hidden_sizes=[trial.suggest_int('hidden_size', 10, 100)]*trial.suggest_int('n_hidden_layers', 1, 3),
                  out_features=out_features, dropout_rate=trial.suggest_float('dropout_rate', 0.0, 0.2))

        # Define the optimizer
        optimizer = torch.optim.Adam(mlp.parameters(), lr=trial.suggest_float('lr', 1e-5, 1e-2))

        # Define the loss function
        loss_fn = nn.MSELoss()

        # Train the model
        mlp = train_fn(mlp, train_loader, val_loader, device=device, optimizer=optimizer, loss_fn=loss_fn)

        # Evaluate the model
        metrics = eval_scores_regression(mlp, val_loader, device)

        return metrics['RMSE']

    # Define the study
    study = optuna.create_study(direction='minimize')

    # Optimize the hyperparameters
    study.optimize(objective, n_trials=10)

    return study.best_params


def optimize_hyperparams_EnsembleModel(backbone, train_fn, train_loader, val_loader, in_features, out_features, device, head_aggregation="mean", get_confidence=False):
    # Define the objective function
    def objective(trial):
        # Define the model
        mlp = EnsembleModel(backbone=backbone, in_features=in_features, hidden_sizes=[trial.suggest_int('hidden_size', 10, 100)]*trial.suggest_int('n_hidden_layers', 1, 3),
                            out_features=out_features, dropout_rate=trial.suggest_float('dropout_rate', 0.0, 0.2), head_aggregation=head_aggregation, get_confidence=get_confidence)

        # Define the optimizer
        optimizer = torch.optim.Adam(mlp.parameters(), lr=trial.suggest_float('lr', 1e-5, 1e-2))

        # Define the loss function
        loss_fn = nn.MSELoss()

        # Train the model
        mlp = train_fn(mlp, train_loader, val_loader, device=device, optimizer=optimizer, loss_fn=loss_fn)

        # Evaluate the model
        metrics = eval_scores_regression(mlp, val_loader, device)

        return metrics['RMSE']

    # Define the study
    study = optuna.create_study(direction='minimize')

    # Optimize the hyperparameters
    study.optimize(objective, n_trials=10)

    return study.best_params


def eval_scores_regression(mlp, test_loader, device):
    # Evaluate the model on the test set
    # Metrics : RMSE and R2
    mlp.eval()

    mse = MeanSquaredError().to(device)
    r2 = R2Score().to(device)

    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        y_hat = mlp(X)
        # y_hat = output[:, 0] + torch.randn_like(output[:, 0]) * torch.sqrt(output[:, 1])
        mse(y_hat, y)
        r2(y_hat, y)

    return {'RMSE': math.sqrt(mse.compute()), 'R2': r2.compute().item()}


# Test on california housing dataset


def get_california_housing_data():
    # Load the dataset
    data = fetch_california_housing()
    X, y = data['data'], data['target']

    # Standardize the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    y = y.reshape(-1, 1)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Convert the dataset to PyTorch tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    # Create the dataloaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    return train_loader, val_loader, test_loader

# ------------------------------


# Define the device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

train_loader, val_loader, test_loader = get_california_housing_data()

# MLP
best_params_mlp = optimize_hyperparams_MLP(train_fn=train_regression, train_loader=train_loader, val_loader=val_loader,
                                           in_features=8, out_features=1, device=device)
mlp = MLP(in_features=8, hidden_sizes=[best_params_mlp['hidden_size']] *
          best_params_mlp['n_hidden_layers'], out_features=1, dropout_rate=best_params_mlp['dropout_rate'])
mlp = train_regression(mlp, train_loader, val_loader, device=device, log_dir="runs/california/mlp")

# Evaluate the model
metrics = {}
hyperparameters = {"mlp": best_params_mlp}

metrics["mlp"] = eval_scores_regression(mlp, test_loader, device)


def eval_ensemble_model(backbone, head_aggregation):
    get_confidence = True if head_aggregation == "weighted" else False
    best_params = optimize_hyperparams_EnsembleModel(backbone=backbone, train_fn=train_regression, train_loader=train_loader, val_loader=val_loader,
                                                     in_features=8, out_features=1, device=device, head_aggregation=head_aggregation, get_confidence=get_confidence)
    hyperparameters[backbone + "_weighted" if head_aggregation == "weighted" else ""] = best_params
    model = EnsembleModel(backbone=backbone, in_features=8, hidden_sizes=[best_params['hidden_size']]*best_params['n_hidden_layers'],
                          out_features=1, dropout_rate=best_params['dropout_rate'], head_aggregation=head_aggregation, get_confidence=get_confidence)
    model = train_regression(model, train_loader, val_loader, device=device, log_dir=f"runs/california/{backbone.__str__()}_{head_aggregation}")

    return eval_scores_regression(model, test_loader, device)


# Non-weighted
metrics["mlpk"] = eval_ensemble_model(MLPk, "mean")
metrics["tabM_naive"] = eval_ensemble_model(TabM_naive, "mean")
metrics["tabM_mini"] = eval_ensemble_model(TabM_mini, "mean")
metrics["tabM"] = eval_ensemble_model(TabM, "mean")

with open("results/california_hyperparameters.json", "w") as f:
    json.dump(hyperparameters, f, indent=4)

with open("results/california.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Weighted
# metrics["mlpk_weighted"] = eval_ensemble_model(MLPk, "weighted")
# metrics["tabM_naive_weighted"] = eval_ensemble_model(TabM_naive, "weighted")
# metrics["tabM_mini_weighted"] = eval_ensemble_model(TabM_mini, "weighted")
# metrics["tabM_weighted"] = eval_ensemble_model(TabM, "weighted")

# with open("results/california_hyperparameters.json", "w") as f:
#     json.dump(hyperparameters, f, indent=4)

# with open("results/california.json", "w") as f:
#     json.dump(metrics, f, indent=4)
