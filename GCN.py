
import torch
from torch._C import parse_ir
import torch.nn.functional as F
from sklearn.manifold import TSNE

import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric import datasets
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv,GATConv # GCN
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

class GCNNet(torch.nn.Module):
    def __init__(self, num_feature, num_label, hiddim, droprate,hidlayers):
        super(GCNNet, self).__init__()
        self.GCN1 = GCNConv(num_feature, hiddim)
        self.GCN = GATConv(hiddim, hiddim)
        self.GCN2 = GCNConv(hiddim, num_label)

        self.hidlayers=hidlayers
        self.dropout = torch.nn.Dropout(p=droprate)

    def forward(self, x, edge_index):
        x=(x-x.mean(dim=0,keepdims=True))/x.std(dim=0,keepdims=True)
        
        x = self.GCN1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        for i in range(self.hidlayers-1):
            x = self.GCN(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.GCN2(x, edge_index)
        return F.log_softmax(x, dim=1)