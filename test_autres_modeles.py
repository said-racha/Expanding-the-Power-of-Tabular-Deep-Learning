import sys
import os


project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, "autres_modeles", "ExcelFormer", "bin"))
sys.path.append(os.path.join(project_root, "autres_modeles", "FT_Transformer", "bin"))
sys.path.append(os.path.join(project_root, "autres_modeles", "T2g_Former", "bin"))

import excel_former
import ft_transformer
import t2g_former

"""
from autres_modeles.ExcelFormer.bin import excel_former
from autres_modeles.FT_Transformer.bin import ft_transformer
from autres_modeles.T2g_Former.bin import t2g_former
"""

import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Chargement des données
data = load_wine()
X = data.data
y = data.target
X = (X - X.mean(axis=0)) / X.std(axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
train_data = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
d_out = 3
num_attributes = X_train.shape[1]

# Modèles


model_config = {
    "d_numerical": num_attributes,
    "categories": [],
    "token_bias": True,
    "n_layers": 3,
    "d_token": 256,
    "n_heads": 32,
    "attention_dropout": 0.3,
    "ffn_dropout": 0.0,
    "residual_dropout": 0.0,
    "prenormalization": True,
    "kv_compression": None,
    "kv_compression_sharing": None,
    "d_out": d_out,
}



ft = ft_transformer.FTTransformer(**model_config, d_ffn_factor=.6, activation="gelu", initialization="xavier")
excel = excel_former.ExcelFormer(**model_config)
t2g = t2g_former.T2GFormer(**model_config, d_ffn_factor=.6, activation="gelu", initialization="xavier")

models = {"FT-Transformer": ft, "Excel-Former": excel, "T2g-Former": t2g}


# Boucle d'entraînement
for name, model in models.items():
    print(f"\n\n{name}\n")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(10): 
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch, None)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

    # Évaluation du modèle
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        predictions = model(X_test_tensor, None)
        preds_classes = predictions.argmax(dim=1)
        print("Test Accuracy:", (preds_classes.numpy() == y_test).mean())
