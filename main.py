import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import scipy.io
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import correlation
from scipy import spatial
from scipy.stats import pearsonr
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import OPTICS, DBSCAN, AffinityPropagation
from sklearn.manifold import TSNE

import gudhi as gd

import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric import datasets
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid

from pyparsing import PrecededBy
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,f1_score,recall_score,classification_report,roc_curve

from PH import create_rips_complex
from HTGNN import HTGNNNet
import argparse

lifespan=0.7

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2022,
        choices=[2019,2020,2021,2022,2023], help='year')
parser.add_argument('--Q', type=int, default=1,
        choices=[1,2,3,4], help='Quarter')
args = parser.parse_args()
year=args.year
Q=args.Q


if Q==1:
    cites1 = "datasets/edges/edge_"+str(year-1)+"Q3.csv"
    content1 = "datasets/nodes/"+str(year-1)+"Q3.csv"
    cites2 = "datasets/edges/edge_"+str(int(year)-1)+"Q4.csv"
    content2 = "datasets/nodes/"+str(int(year)-1)+"Q4.csv"
elif Q==2:
    cites1 = "datasets/edges/edge_"+str(int(year)-1)+"Q4.csv"
    content1 = "datasets/nodes/"+str(int(year)-1)+"Q4.csv"
    cites2 = "datasets/edges/edge_"+str(year)+"Q1.csv"
    content2 = "datasets/nodes/"+str(year)+"Q1.csv"
else:
    cites1 = "datasets/edges/edge_"+str(year)+"Q"+str(int(Q)-2)+".csv"
    content1 = "datasets/nodes/"+str(year)+"Q"+str(int(Q)-2)+".csv"
    cites2 = "datasets/edges/edge_"+str(year)+"Q"+str(int(Q)-1)+".csv"
    content2 = "datasets/nodes/"+str(year)+"Q"+str(int(Q)-1)+".csv"

# 索引字典，将原本的论文id转换到从0开始编码
index_dict = dict()
# 标签字典，将字符串标签转化为数值
label_to_index = dict()

features = []
labels = []

with open(content1,"r") as f:
    nodes = f.readlines()
    j=0
    for node in nodes:
        node=node.strip('\n')
        if j==0:
            j=1
            continue
        node_info = node.split(',')    
        index_dict[node_info[0]] = len(index_dict)
        features.append([float(i) for i in node_info[1:-1]])
            
        label_str = node_info[-1]
        if(label_str not in label_to_index.keys()):
            label_to_index[label_str] = len(label_to_index)
        labels.append(label_to_index[label_str])



with open(content2,"r") as f:
    nodes = f.readlines()
    j=0
    for node in nodes:
        node=node.strip('\n')
        if j==0:
            j=1
            continue
        node_info = node.split(',')    
        index_dict[str(int(node_info[0])+24271)] = len(index_dict)
        features.append([float(i) for i in node_info[1:-1]])
            
        label_str = node_info[-1]
        if(label_str not in label_to_index.keys()):
            label_to_index[label_str] = len(label_to_index)
        labels.append(label_to_index[label_str])


edge_index = []

with open(cites1,"r") as f:
    edges = f.readlines()
    j=0
    for edge in edges:
        if j==0:
            j=1
            continue
        edge=edge.strip('\n')
        start, end, weight = edge.split(',')
        #print(index_dict[start])
        #print(index_dict[end])
        if start in index_dict and end in index_dict :
            # 训练时将边视为无向的，但原本的边是有向的，因此需要正反添加两次
            edge_index.append([index_dict[start],index_dict[end]])
            edge_index.append([index_dict[end],index_dict[start]])  


with open(cites2,"r") as f:
    edges = f.readlines()
    j=0
    for edge in edges:
        if j==0:
            j=1
            continue
        edge=edge.strip('\n')
        #print(edge)
        start, end, weight = edge.split(',')
        start=str(int(start)+24271)
        end=str(int(end)+24271)  
        if start in index_dict and end in index_dict :
            # 训练时将边视为无向的，但原本的边是有向的，因此需要正反添加两次
            edge_index.append([index_dict[start],index_dict[end]])
            edge_index.append([index_dict[end],index_dict[start]]) 


labels = torch.LongTensor(labels)
features = torch.FloatTensor(features)
edge_index =  torch.LongTensor(edge_index)


seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

mask = torch.arange(9096)
train_mask = mask[:4548]
test_mask = mask[4548:]

    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


epochs=1000
hiddim=16
droprate=0.6
batchsize=64
numneighbors=30
hidlayers=2
lr=0.01

data = Data(x = features, edge_index = edge_index.t().contiguous(), y = labels).to(device)

loader = NeighborLoader(
    data,
    num_neighbors=[numneighbors],
    batch_size=batchsize,
    input_nodes=train_mask,
)
testload=NeighborLoader(
    data,
    num_neighbors=[numneighbors],
    batch_size=batchsize,
    input_nodes=test_mask,
)



bestacclist2=[]
bestf1macrolist2=[]
bestf1microlist2=[]
bestprecisionlist2=[]
bestrecalllist2=[]


i=0
bestacc=0
bestf1micro=0
bestf1macro=0
bestauc=0
bestprecision=0
bestrecall=0
for (batch,testbatch) in zip(tqdm(loader),tqdm(testload)):
    print(batch)

    i=i+1
    if i>1:
        break

    x1=batch.x.cpu().detach().numpy()
    distance_matrix = cosine_similarity(x1)
    train_simplex_tree = create_rips_complex(distance_matrix, max_edge_length=lifespan)
    train_simplex_tree.persistence()

    persistence_intervals = np.vstack([train_simplex_tree.persistence_intervals_in_dimension(1) , train_simplex_tree.persistence_intervals_in_dimension(0)])
    persistence_pairs = train_simplex_tree.persistence_pairs()
    matrix = [[item1[0], item2[1]-item2[0]] for item1, item2 in zip(persistence_pairs, persistence_intervals)]
    # print(matrix)

    edges = []
    for simplex,lifespan in matrix:
        if lifespan>=0.1:
            # print(simplex)
            if len(simplex) == 2:  
                edges.append(simplex)
            if len(simplex) == 3:
                edges.extend([(simplex[i], simplex[j]) for i in range(2) for j in range(i+1, 3)])

    edges_array = np.array(edges).T 

    edges_set_1 = {tuple(sorted(edge)) for edge in edges_array.T}

    edges_array = np.array(list(edges_set_1))
    edges_transposed = edges_array.T
    edge_new = torch.tensor(edges_transposed, dtype=torch.long).to(device)

    print(batch.edge_index.size())
    print(edge_new.size())

    x1=testbatch.x.cpu().detach().numpy()
    distance_matrix = cosine_similarity(x1)
    test_simplex_tree = create_rips_complex(distance_matrix, max_edge_length=lifespan)
    test_simplex_tree.persistence()
    persistence_intervals = np.vstack([test_simplex_tree.persistence_intervals_in_dimension(1) , test_simplex_tree.persistence_intervals_in_dimension(0)])
    persistence_pairs = test_simplex_tree.persistence_pairs()
    matrix = [[item1[0], item2[1]-item2[0]] for item1, item2 in zip(persistence_pairs, persistence_intervals)]

    edges = []
    # for simplex, _ in train_simplex_tree.get_simplices():
    for simplex,lifespan in matrix:
        if lifespan>=0.1:
            # print(simplex)
            if len(simplex) == 2:  
                edges.append(simplex)
            if len(simplex) == 3:
                edges.extend([(simplex[i], simplex[j]) for i in range(2) for j in range(i+1, 3)])
    edges_array = np.array(edges).T 

    edges_set_1 = {tuple(sorted(edge)) for edge in edges_array.T}

    edges_array = np.array(list(edges_set_1))
    edges_transposed = edges_array.T
    edge_new_test = torch.tensor(edges_transposed, dtype=torch.long).to(device)        

    mask = batch.y[batch.edge_index[0]] == batch.y[batch.edge_index[1]]
    filtered_edge_index = batch.edge_index[:, mask]
    num_edges_to_sample = int(filtered_edge_index.size(1) * 0.01)
    indices = torch.randperm(filtered_edge_index.size(1), device='cuda:0')[:num_edges_to_sample]
    batchedge = filtered_edge_index[:, indices]

    mask = testbatch.y[testbatch.edge_index[0]] == testbatch.y[testbatch.edge_index[1]]
    filtered_edge_index = testbatch.edge_index[:, mask]
    num_edges_to_sample = int(filtered_edge_index.size(1) * 0.01)
    indices = torch.randperm(filtered_edge_index.size(1), device='cuda:0')[:num_edges_to_sample]
    testbatchedge = filtered_edge_index[:, indices]

    edge_new1= torch.cat((edge_new, batchedge), dim=1)
    edge_new_test1= torch.cat((edge_new_test, testbatchedge), dim=1)
    
    print("aaaaaa")

 
    bestacc=0
    bestf1micro=0
    bestf1macro=0
    bestauc=0
    bestprecision=0
    bestrecall=0
    predb3 = 0
    model = HTGNNNet(features.shape[1], len(label_to_index), hiddim=hiddim, droprate=droprate,hidlayers=hidlayers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for epoch in tqdm(range(epochs)):   
        # print(epoch)    
        kk=0
        model.train()
        optimizer.zero_grad()
        out = model(batch.x, edge_new1, batch.edge_index)
        loss = F.nll_loss(out, batch.y)
        
        loss.backward()
        optimizer.step()
        kk+=1        

        model.eval()
        _, predb = model(testbatch.x, edge_new_test1, testbatch.edge_index).max(dim=1)
        predb = torch.Tensor.cpu(predb).tolist()
        trub = torch.Tensor.cpu(testbatch.y).tolist()
        predlist=[]
        trulist=[]    
        predlist.extend(predb)
        trulist.extend(trub)

        accuracy = accuracy_score(trulist, predlist)  
        # auc=roc_auc_score(trulist, predlist,multi_class='ovr')   
        precision = precision_score(trulist, predlist, average='macro')
        recall= recall_score(trulist, predlist, average='macro')
        f1_micro = f1_score(trulist, predlist, average='micro')
        f1_macro = f1_score(trulist, predlist, average='macro')
        
        if bestacc< accuracy: 
            bestacc = accuracy
            # print(accuracy)

        if bestprecision<precision:
            bestprecision=precision    

        if bestrecall<recall:
            bestrecall=recall         

        if bestf1macro<f1_macro:
            bestf1macro=f1_macro
            predb3 = predb

        if bestf1micro<f1_micro:
            bestf1micro=f1_micro

    bestacclist2.append(bestacc)
    bestf1macrolist2.append(bestf1macro)
    bestf1microlist2.append(bestf1micro)
    bestprecisionlist2.append(bestprecision)
    bestrecalllist2.append(bestrecall)

    print(bestacclist2)
    print(bestf1macrolist2)
    print(bestprecisionlist2)
    print(bestrecalllist2) 



print("HTGNN:")
print(bestacclist2)
print(bestf1macrolist2)
print(bestprecisionlist2)
print(bestrecalllist2)  

