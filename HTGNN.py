
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

class HTGNNNet(torch.nn.Module):
    def __init__(self, num_feature, num_label, hiddim, droprate, hidlayers):
        super(HTGNNNet, self).__init__()

        # 定义 GCN 层，用于两种不同的边集
        self.GCN_q = GCNConv(num_feature, hiddim)
        self.GCN_p = GCNConv(num_feature, hiddim)
        self.GCN1_q  = GCNConv(hiddim, hiddim)
        self.GCN1_p = GCNConv(hiddim, hiddim)
        self.GCN2_q  = GCNConv(hiddim, num_label)
        self.GCN2_p = GCNConv(hiddim, num_label)

        # 隐藏层数和 Dropout
        self.hidlayers = hidlayers
        self.dropout = torch.nn.Dropout(p=droprate)

        # 自定义的 α 系数
        self.alpha_q = 0.7
        self.alpha_p = 0.3

    def forward(self, x, edge_index_q, edge_index_p):
        # 标准化输入特征
        x = (x - x.mean(dim=0, keepdims=True)) / x.std(dim=0, keepdims=True)

        term_q = self.alpha_q * self.dropout(F.relu( self.GCN_q(x, edge_index_q)))
        term_p = self.alpha_p * self.dropout(F.relu( self.GCN_p(x, edge_index_p)))
        x = term_q + term_p

        for i in range(self.hidlayers-1):
            term_q = self.alpha_q * self.dropout(F.relu( self.GCN1_q(x, edge_index_q)))
            term_p = self.alpha_p * self.dropout(F.relu( self.GCN1_p(x, edge_index_p)))
            x = term_q + term_p           
        x = self.alpha_q * self.GCN2_q(x, edge_index_q) + self.alpha_p * self.GCN2_p(x, edge_index_p)

        return F.log_softmax(x, dim=1)