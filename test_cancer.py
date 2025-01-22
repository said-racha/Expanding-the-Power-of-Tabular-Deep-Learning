# Classification binaire : `breast_cancer`

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tabm_raph as raph

def get_cancer_data(split=.2, batch_size=32, seed=42):

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=split, random_state=seed)
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    train_loader =  DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



def train_cancer(net, train_loader, test_loader, log_dir, criterion = nn.BCEWithLogitsLoss(), lr = 1e-3, nb_iter = 50):

    optim = torch.optim.AdamW(net.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)
    
    for epoch in tqdm(range(nb_iter)):
        
        net.train()
        losses = []
        train_correct = 0
        train_total = 0
        
        for x,y in train_loader:
            preds = net(x).view(-1)
            loss = criterion(preds,y)
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

            for x,y in test_loader: 
                preds = net(x).view(-1)
                loss = criterion(preds,y)
                test_losses.append(loss)

                # Compute accuracy
                preds_binary = (torch.sigmoid(preds) > 0.5).int()
                test_correct += (preds_binary == y).sum()
                test_total += y.size(0)

            writer.add_scalar("Loss/test", torch.tensor(test_losses).mean(), epoch)
            writer.add_scalar("Accuracy/test", test_correct / test_total, epoch)



train_loader, test_loader = get_cancer_data()

tabM_naive = raph.TabM_Naive([30, 64, 32, 16, 1])
tabM = raph.TabM([30, 64, 32, 16, 1])
simple_MLP = nn.Sequential(nn.Linear(30, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
MLP_k = raph.MLP_k(simple_MLP)


train_cancer(tabM_naive, train_loader, test_loader, 'runs/tabM_naive')
train_cancer(simple_MLP, train_loader, test_loader, 'runs/MLP')
train_cancer(tabM, train_loader, test_loader, 'runs/tabM')
train_cancer(MLP_k, train_loader, test_loader, 'runs/MLP_k')