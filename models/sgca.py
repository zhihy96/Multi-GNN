import torch
import torch.nn.functional as F
from torch.nn import ReLU, Sequential as Seq, Linear
from torch_geometric.nn import global_add_pool,ARMAConv,SGConv
from torch_geometric.data import DataLoader, Batch

class SGCA(torch.nn.Module):
    def __init__(self, n_features, n_outputs, dim=100):
        super(SGCA, self).__init__()
        # 
        self.sgc1 = SGConv(n_features,dim)
        self.sgc2 = SGConv(dim,dim)
        self.bn = torch.nn.BatchNorm1d(dim)
        self.armaconv = ARMAConv(dim, dim)
         
        # the Fully Connected Layer
        self.fc1 = Linear(dim, 2*dim)
        self.fc2 = Linear(2*dim, 3*dim)
        self.fc3 = Linear(3*dim, 2*dim)
        self.fc4 = Linear(2*dim, 1)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 
        x = F.relu(self.sgc1(x, edge_index))
        x = F.relu(self.sgc2(x, edge_index))
        #
        x = self.bn(x)
        #
        x = self.armaconv(x, edge_index)
        
        # Fully Connected Layer
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
