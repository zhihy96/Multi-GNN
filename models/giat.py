import torch
import torch.nn.functional as F
from torch.nn import ReLU, Sequential as Seq, Linear
from torch_geometric.nn import global_add_pool,GINConv,GATConv 
from torch_geometric.data import DataLoader, Batch

class GIAT(torch.nn.Module):
    def __init__(self, n_features, n_outputs, dim=100):
        super(GIAT, self).__init__()
        # the  
        nn1 = Seq(Linear(n_features, 2*dim), ReLU(), Linear(2*dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.conv2 = GATConv(dim, dim,heads=4)
        
        # Preparation of the Fully Connected Layer
        self.fc1 = Linear(4*dim, 3*dim)
        self.fc2 = Linear(3*dim, 2*dim)
        self.fc3 = Linear(2*dim, dim)
        self.fc4 = Linear(dim, 1)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Graph Isomorphism Convolutional Layer
        x = F.relu(self.conv1(x, edge_index))
        # 
        x = self.bn1(x)
        #        
        x = self.conv2(x, edge_index)

        # Fully Connected Layer
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
