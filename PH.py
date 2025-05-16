
import scipy.io
import networkx as nx
import numpy as np

import os, torch, numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import networkx as nx
import torch, os, numpy as np, scipy.sparse as sp
from scipy.sparse import coo_matrix,csr_matrix
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
from tqdm import tqdm
import math
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from itertools import chain
from sklearn.cluster import OPTICS,DBSCAN,AffinityPropagation
import matplotlib.pyplot as plt
from scipy.spatial.distance import correlation
import gudhi as gd
import numpy as np


def all_same(tensor):
    # if tensor.nelement() == 0:  # 检查张量是否为空
    #     return 1
    return 0 if torch.any(tensor != tensor[0]) else 1


def create_rips_complex(feature_matrix, max_edge_length):
    rips_complex = gd.RipsComplex(points=feature_matrix, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    return simplex_tree

def compute_persistence_diagram(simplex_tree):
    return simplex_tree.persistence()

def compute_average_proportion(simplex_tree, y):
    proportion = [all_same(y[simplex[0]]) for simplex in simplex_tree.get_skeleton(2) if len(simplex[0])>1]
    return np.mean(proportion) if proportion else 0

def compute_average_lifetime(persistence_diagram):
    lifetimes = [death - birth for _,(birth, death) in persistence_diagram if death != float('inf')]
    return np.mean(lifetimes) if lifetimes else 0

def compute_average_density(persistence_diagram):
    density = [ (death + birth)/2 for _,(birth, death) in persistence_diagram if death != float('inf')]
    return np.mean(density) if density else 0

def compute_simplex_length(simplex_tree):
    distribution = [len(simplex[0]) for simplex in simplex_tree.get_skeleton(2)]
    return distribution 

def compute_pred_proportion(simplex_tree, y, pred):
    ysameproportion = [[simplex[0],y[simplex[0]],pred[simplex[0]]] for simplex in simplex_tree.get_skeleton(2) if len(simplex[0])>1 and all_same(y[simplex[0]]) and torch.equal(y[simplex[0]],pred[simplex[0]])==0]
    predsameproportion = [[simplex[0],y[simplex[0]],pred[simplex[0]]] for simplex in simplex_tree.get_skeleton(2) if len(simplex[0])>1 and all_same(pred[simplex[0]]) and torch.equal(y[simplex[0]],pred[simplex[0]])==0]
    nosameproportion = [[simplex[0],y[simplex[0]],pred[simplex[0]]] for simplex in simplex_tree.get_skeleton(2) if len(simplex[0])>1 and torch.equal(y[simplex[0]],pred[simplex[0]])==0]
    return ysameproportion, predsameproportion, nosameproportion

def compute_edge_indicator_vector(edges, num_nodes):
    indicator = np.zeros((num_nodes, num_nodes))
    for edge in edges:
        indicator[edge[0], edge[1]] = 1
        indicator[edge[1], edge[0]] = 1  # 无向图，对称
    return indicator.flatten()

def linear_mmd(x, y):
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)
    mmd = np.linalg.norm(mean_x - mean_y)
    return mmd

def jaccard_similarity(edges1, edges2):
    set1 = {tuple(sorted(edge)) for edge in edges1.T}
    set2 = {tuple(sorted(edge)) for edge in edges2.T}

    print(set1)
    print(set2)
    
    # 计算交集和并集
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # 计算Jaccard相似性
    jaccard_index = len(intersection) / len(union)
    return jaccard_index
