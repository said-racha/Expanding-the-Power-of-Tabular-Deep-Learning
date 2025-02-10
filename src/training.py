import warnings
import time
import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, r2_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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
