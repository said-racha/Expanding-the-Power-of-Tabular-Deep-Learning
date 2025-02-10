import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_iris
from ucimlrepo import fetch_ucirepo


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
