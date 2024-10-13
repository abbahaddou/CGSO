import pickle
import argparse
import numpy as np
import os.path as osp
import scipy.sparse as sp
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils import degree
from pathlib import Path
import networkx as nx
from numpy import dot
import torch
import community.community_louvain as community_louvain
import wandb
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import igraph as ig
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import sys

########################################################################################
# Parse arguments 
########################################################################################

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0,help='Set CUDA device number; if set to -1, disables cuda.')    
parser.add_argument('--use_wandb', type= bool,default = False , choices=[True, False])
args = parser.parse_args()
device = torch.device('cuda:'+str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
                        

list_ami = []
list_ari = []
for pam_graph_id in range(200) :
    G = pickle.load(open('./Spectral_clustering_BA/synthetic_graphs/G_{}.pickle'.format(pam_graph_id), 'rb'))
    n = len(G.nodes)
    g = ig.Graph.TupleList(G.edges(), directed=False)
    wtrap = g.community_walktrap(steps = 4)
    clust = wtrap.as_clustering()
    len_clusters = len(clust)

    clusters = [-1 for i in range(n)]
    iii = 0
    for i_ in range(len_clusters) :
        c = clust[i_]
        for v in c :
            clusters[v] = iii
        iii += 1 
    labels = [0 for ii in range(100)]+ [1 for ii in range(100)]+ [2 for ii in range(100)]

    ami = adjusted_mutual_info_score( labels, clusters,average_method = 'arithmetic')
    ari = adjusted_rand_score( labels , clusters)
    list_ami.append(ami)
    list_ari.append(ari)

print('ami : {} +- {}'.format(np.mean(list_ami) , np.std(list_ami)) )
print('ari : {} +- {}'.format(np.mean(list_ari) , np.std(list_ari)) )
