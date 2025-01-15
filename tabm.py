import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




class LinearBE(nn.Module): # BatchEnsemble layer
    def __init__(self, dim_in:int, dim_out:int, k=32):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.k = k
        self.R = nn.Parameter(torch.Tensor(k, dim_in))
        self.W = nn.Parameter(torch.Tensor(dim_in, dim_out))
        self.S = nn.Parameter(torch.Tensor(k, dim_out))
        self.B = nn.Parameter(torch.Tensor(k, dim_out))

        # "ri and si are randomly initialized with ±1 to ensure diversity of the k linear layers" (l.171-172)
        nn.init.uniform_(self.R, -1, 1)
        nn.init.uniform_(self.S, -1, 1)

        nn.init.normal_(self.W)
        nn.init.normal_(self.B)


    def forward(self, X):
        """
        "LinearBE(X) = ( (X * R) W) * S + B" (l.180)

        X has shape (batch_size, k, dim_in)
        R has shape (k, dim_in)
        W has shape (dim_in, dim_out)
        S has shape (k, dim_out)
        B has shape (k, dim_out)
        """

        # Element-wise multiplication of X and R
        output = torch.mul(X, self.R) # (batch_size, k, dim_in)

        # Matrix multiplication with W
        output = torch.einsum("bki,io->bko", output, self.W)  # (batch_size, k, dim_out)

        # Element-wise multiplication with S and addition of B
        output = output * self.S + self.B  # (batch_size, k, dim_out)

        return output
    


class TabM_Naive(nn.Module):
    def __init__(self, layers_shapes:list, k=32):
        super().__init__()

        self.k = k

        self.layers = torch.nn.ModuleList()

        # "applying BatchEnsemble to all linear layers" (l.200)
        for i in range(len(layers_shapes)-1):
            self.layers.append(LinearBE(layers_shapes[i], layers_shapes[i+1], k))
            self.layers.append(torch.nn.ReLU())
        
        # "fully non-shared prediction heads" (l.201)
        self.pred_heads = nn.ModuleList([nn.Linear(layers_shapes[-1], 1) for _ in range(k)])

    
    def forward(self, x):
        
        """
        "X represents k representations of the same input object x" (l.181)
        so, x has shape (batch, dim)
        and X has shape (batch, k, dim)
        """
        X = x.unsqueeze(1).repeat(1, self.k, 1)

        for layer in self.layers:
            X = layer(X)
        
        # predictions
        preds = torch.cat([head(X[:,i]) for i, head in enumerate(self.pred_heads)], dim=1)
        return preds.mean(dim=1)





####### Training

BATCH_SIZE = 32

def train_cancer(net, train_loader, test_loader, log_dir, criterion = nn.BCEWithLogitsLoss(), lr = 1e-3, nb_iter = 50):

    optim = torch.optim.Adam(net.parameters(), lr=lr)
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



# Classification : `breast_cancer`

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
train_loader =  DataLoader(list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=BATCH_SIZE, shuffle=False)


tabM_naive = TabM_Naive([X_train.shape[1], 64, 32, 16])
simple_MLP = nn.Sequential(nn.Linear(X_train.shape[1], 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
train_cancer(tabM_naive, train_loader, test_loader, 'runs/tabM_naive_breast_cancer')
train_cancer(simple_MLP, train_loader, test_loader, 'runs/MLP')
