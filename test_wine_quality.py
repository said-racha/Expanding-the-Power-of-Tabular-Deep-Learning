from ucimlrepo import fetch_ucirepo

import torch

from sklearn.model_selection import train_test_split

from test_wine import train_multiclass_classification
import tabm_luc as luc


def get_quality_wine_data(split=.2, batch_size=32, seed=42):
    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets

    # metadata
    # print(wine_quality.metadata)

    # variable information
    # print(wine_quality.variables)

    # Normalize
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train.to_numpy()).float(), torch.tensor(y_train.to_numpy()).long()),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_test.to_numpy()).float(), torch.tensor(y_test.to_numpy()).long()),
        batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Classification : `wine`
    BATCH_SIZE = 32

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    train_loader, test_loader = get_quality_wine_data(split=.2, batch_size=BATCH_SIZE, seed=42)

    hidden_sizes = [64, 64]

    in_features = out_features = 11

    # ===== Classic Models =====

    mlp = luc.MLP(in_features, hidden_sizes, out_features, dropout_rate=0).to(device)
    mlpk = luc.EnsembleModel(luc.MLPk, in_features, hidden_sizes, out_features, dropout_rate=0, get_confidence=True).to(device)
    tabM_naive = luc.EnsembleModel(luc.TabM_naive, in_features, hidden_sizes, out_features, dropout_rate=0, get_confidence=True).to(device)
    tabM_mini = luc.EnsembleModel(luc.TabM_mini, in_features, hidden_sizes, out_features, dropout_rate=0, get_confidence=True).to(device)
    tabM = luc.EnsembleModel(luc.TabM, in_features, hidden_sizes, out_features, dropout_rate=0, get_confidence=True).to(device)

    train_multiclass_classification(mlp, train_loader, test_loader, "runs/wine_quality/luc/MLP", device)
    train_multiclass_classification(mlpk, train_loader, test_loader, "runs/wine_quality/luc/MLPk", device, log_confidence=True)
    train_multiclass_classification(tabM_naive, train_loader, test_loader, "runs/wine_quality/luc/TabM_naive", device, log_confidence=True)
    train_multiclass_classification(tabM_mini, train_loader, test_loader, "runs/wine_quality/luc/TabM_mini", device, log_confidence=True)
    train_multiclass_classification(tabM, train_loader, test_loader, "runs/wine_quality/luc/TabM", device, log_confidence=True)

    # ===== Weighted Models =====

    mlpk_weighted = luc.EnsembleModel(luc.MLPk, in_features, hidden_sizes, out_features, dropout_rate=0,
                                      get_confidence=True, head_aggregation="weighted").to(device)
    tabM_naive_weighted = luc.EnsembleModel(luc.TabM_naive, in_features, hidden_sizes, out_features, dropout_rate=0,
                                            get_confidence=True, head_aggregation="weighted").to(device)
    tabM_mini_weighted = luc.EnsembleModel(luc.TabM_mini, in_features, hidden_sizes, out_features, dropout_rate=0,
                                           get_confidence=True, head_aggregation="weighted").to(device)
    tabM_weighted = luc.EnsembleModel(luc.TabM, in_features, hidden_sizes, out_features, dropout_rate=0,
                                      get_confidence=True, head_aggregation="weighted").to(device)

    train_multiclass_classification(mlpk_weighted, train_loader, test_loader, "runs/wine_quality/luc/MLPk_weighted", device, log_confidence=True)
    train_multiclass_classification(tabM_naive_weighted, train_loader, test_loader, "runs/wine_quality/luc/TabM_naive_weighted", device, log_confidence=True)
    train_multiclass_classification(tabM_mini_weighted, train_loader, test_loader, "runs/wine_quality/luc/TabM_mini_weighted", device, log_confidence=True)
    train_multiclass_classification(tabM_weighted, train_loader, test_loader, "runs/wine_quality/luc/TabM_weighted", device, log_confidence=True)
