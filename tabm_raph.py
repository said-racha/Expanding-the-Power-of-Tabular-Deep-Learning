import torch
import torch.nn as nn

from copy import deepcopy




class LinearBE(nn.Module): # BatchEnsemble layer
    def __init__(self, dim_in:int, dim_out:int, k=32, initR="random"):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.k = k
        self.R = nn.Parameter(torch.Tensor(k, dim_in))
        self.W = nn.Parameter(torch.Tensor(dim_in, dim_out))
        self.S = nn.Parameter(torch.Tensor(k, dim_out))
        self.B = nn.Parameter(torch.Tensor(k, dim_out))

        if initR == "random": # TabM naive
            # "randomly initialized with ±1 to ensure diversity of the k linear layers" (l.171-172)
            nn.init.uniform_(self.R, -1, 1)

        elif initR == "ones": # TabM
            nn.init.ones_(self.R)
        
        else:
            raise ValueError("init should be 'random' or 'ones'")

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
        for i in range(len(layers_shapes)-2):
            self.layers.append(LinearBE(layers_shapes[i], layers_shapes[i+1], k))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(0.1))
        
        # "fully non-shared prediction heads" (l.201)
        self.pred_heads = nn.ModuleList([nn.Linear(layers_shapes[-2], layers_shapes[-1]) for _ in range(k)])

    
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
        preds = torch.stack([head(X[:,i]) for i, head in enumerate(self.pred_heads)], dim=1)
        return preds.mean(dim=1)


class TabM(nn.Module):
    def __init__(self, layers_shapes:list, k=32):
        super().__init__()

        self.k = k

        self.layers = torch.nn.ModuleList([LinearBE(layers_shapes[0], layers_shapes[1], k, initR="ones"),
                                          torch.nn.ReLU(),
                                          torch.nn.Dropout(0.1)])

        for i in range(1,len(layers_shapes)-2):
            self.layers.append(LinearBE(layers_shapes[i], layers_shapes[i+1], k))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(0.1))
        
        self.pred_heads = nn.ModuleList([nn.Linear(layers_shapes[-2], layers_shapes[-1]) for _ in range(k)])

    
    def forward(self, x):
        X = x.unsqueeze(1).repeat(1, self.k, 1)

        for layer in self.layers:
            X = layer(X)
        
        # predictions
        preds = torch.stack([head(X[:,i]) for i, head in enumerate(self.pred_heads)], dim=1)
        return preds.mean(dim=1)


class MLP_k(nn.Module):
    def __init__(self, MLP, k=32):
        super().__init__()
        self.k = k
        self.MLPs = nn.ModuleList([deepcopy(MLP) for _ in range(k)])
    
    def forward(self, x):
        return torch.stack([MLP(x) for MLP in self.MLPs], dim=1).mean(dim=1)


