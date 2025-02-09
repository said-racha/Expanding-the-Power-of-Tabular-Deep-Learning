import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import tabm_luc as luc
import tabm_racha as racha
import tabm_raph as raph

import plotly.graph_objects as go


def train_multiclass_classification(net, train_loader, test_loader, log_dir, device, criterion=nn.CrossEntropyLoss(), lr=1e-3, nb_iter=20, verbose=True, log_confidence=False):
    optim = torch.optim.AdamW(net.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in tqdm(range(nb_iter), desc="Training", unit="ep"):
        # Training

        net.train()
        losses = []
        train_correct = 0
        train_total = 0
        
        if log_confidence:
            confidences = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.reshape(-1).to(device)
            
            if log_confidence:
                yhat, confidence = net(x)
                confidences.append(confidence.mean())
            else:
                yhat = net(x)

            loss = criterion(yhat, y)
            losses.append(loss)

            # Compute accuracy
            preds_classes = torch.argmax(yhat, dim=1)
            train_correct += (preds_classes == y).sum()
            train_total += y.size(0)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optim.step()

        if verbose:
            writer.add_scalar("Loss/train", torch.tensor(losses).mean(), epoch)
            writer.add_scalar("Accuracy/train", train_correct / train_total, epoch)
            if log_confidence:
                writer.add_scalar("Confidence/train", torch.tensor(confidences).mean(), epoch)

        net.eval()
        with torch.no_grad():
            test_losses = []
            test_correct = 0
            test_total = 0
            
            if log_confidence:
                confidences = []

            for x, y in test_loader:
                x = x.to(device)
                y = y.reshape(-1).to(device)
                
                
                if log_confidence:
                    yhat, confidence = net(x)
                    confidences.append(confidence.mean())
                else:
                    yhat = net(x)
                loss = criterion(yhat, y)
                test_losses.append(loss)

                # Compute accuracy
                preds_classes = torch.argmax(yhat, dim=1)
                test_correct += (preds_classes == y).sum()
                test_total += y.size(0)

            if verbose:
                writer.add_scalar("Loss/test", torch.tensor(test_losses).mean(), epoch)
                writer.add_scalar("Accuracy/test", test_correct / test_total, epoch)
                if log_confidence:
                    writer.add_scalar("Confidence/test", torch.tensor(confidences).mean(), epoch)
            
            
def get_wine_data(split=.2, batch_size=32, seed=42):
    """
    Loads the wine dataset into dataloaders
    """
    data = load_wine()
    X = data.data
    y = data.target

    # Normalize
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
    
    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()),
        batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader, torch.tensor(X_train).float()

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
    
    train_loader, test_loader, X_train = get_wine_data(split=.2, batch_size=BATCH_SIZE, seed=42)
    
    hidden_sizes = [64, 64]
    
    
    # ===== Luc =====
    """
    mlp = luc.MLP(13, hidden_sizes, 3, dropout_rate=0).to(device)
    mlpk = luc.EnsembleModel(luc.MLPk, 13, hidden_sizes, 3, dropout_rate=0, get_confidence=True).to(device)
    tabM_naive = luc.EnsembleModel(luc.TabM_naive, 13, hidden_sizes, 3, dropout_rate=0, get_confidence=True).to(device)
    tabM_mini = luc.EnsembleModel(luc.TabM_mini, 13, hidden_sizes, 3, dropout_rate=0, get_confidence=True).to(device)
    tabM = luc.EnsembleModel(luc.TabM, 13, hidden_sizes, 3, dropout_rate=0, get_confidence=True).to(device)
    
    train_multiclass_classification(mlp, train_loader, test_loader, "runs/wine/MLP", device)
    train_multiclass_classification(mlpk, train_loader, test_loader, "runs/wine/MLPk", device, log_confidence=True)
    train_multiclass_classification(tabM_naive, train_loader, test_loader, "runs/wine/TabM_naive", device, log_confidence=True)
    train_multiclass_classification(tabM_mini, train_loader, test_loader, "runs/wine/TabM_mini", device, log_confidence=True)
    train_multiclass_classification(tabM, train_loader, test_loader, "runs/wine/TabM", device, log_confidence=True)
    
    mlpk_weighted = luc.EnsembleModel(luc.MLPk, 13, hidden_sizes, 3, dropout_rate=0, head_aggregation="weighted", get_confidence=True).to(device)
    tabM_naive_weighted = luc.EnsembleModel(luc.TabM_naive, 13, hidden_sizes, 3, dropout_rate=0, head_aggregation="weighted", get_confidence=True).to(device)
    tabM_mini_weighted = luc.EnsembleModel(luc.TabM_mini, 13, hidden_sizes, 3, dropout_rate=0, head_aggregation="weighted", get_confidence=True).to(device)
    tabM_weighted = luc.EnsembleModel(luc.TabM, 13, hidden_sizes, 3, dropout_rate=0, head_aggregation="weighted", get_confidence=True).to(device)
    
    train_multiclass_classification(mlpk_weighted, train_loader, test_loader, "runs/wine/MLPk_weighted", device, log_confidence=True)
    train_multiclass_classification(tabM_naive_weighted, train_loader, test_loader, "runs/wine/TabM_naive_weighted", device, log_confidence=True)
    train_multiclass_classification(tabM_mini_weighted, train_loader, test_loader, "runs/wine/TabM_mini_weighted", device, log_confidence=True)
    train_multiclass_classification(tabM_weighted, train_loader, test_loader, "runs/wine/TabM_weighted", device, log_confidence=True)
    """
    # ===== Racha =====
    
    # mlp = racha.SimpleMLP(13, hidden_sizes, 3).to(device)
    # tabM = racha.TabM(13, hidden_sizes, 3, 32).to(device)
    
    # train_multiclass_classification(mlp, train_loader, test_loader, "runs/wine/racha/MLP", device)
    # train_multiclass_classification(tabM, train_loader, test_loader, "runs/wine/racha/TabM", device)
    
    # ===== Raphaël =====
    
    # mlp = nn.Sequential(nn.Linear(13, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3)).to(device)
    # mlpk = raph.MLP_k(mlp).to(device)
    # tabM_naive = raph.TabM_Naive([13, 64, 32, 16, 3]).to(device)
    tabM = raph.TabM([13, 64, 32, 16, 3]).to(device)
    nonLinearTabM = raph.NonLinearTabM([13, 64, 32, 16, 3]).to(device)
    
    # train_multiclass_classification(mlp, train_loader, test_loader, "runs/wine/raph/MLP", device)
    # train_multiclass_classification(mlpk, train_loader, test_loader, "runs/wine/raph/MLPk", device)
    # train_multiclass_classification(tabM_naive, train_loader, test_loader, "runs/wine/raph/TabM_naive", device)
    train_multiclass_classification(tabM, train_loader, test_loader, "runs/wine/raph/TabM", device)
    train_multiclass_classification(nonLinearTabM, train_loader, test_loader, "runs/wine/raph/NonLinTabM", device, lr=1e-2)
    
    # tabM_ple = raph.TabM_with_PLE([130, 64, 3]).to(device)
    # tabM_ple.fit_bins(X_train) # initialisation des bins
    # train_multiclass_classification(tabM_ple, train_loader, test_loader, "runs/wine/raph/TabM_PLE", device)
