import pickle
import argparse
import numpy as np
import os.path as osp
import scipy.sparse as sp
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils import degree
from pathlib import Path
import networkx as nx
from node2vec import Node2Vec
from numpy import dot
import torch
import wandb
from torch_geometric.utils.convert import from_scipy_sparse_matrix
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

    adj = nx.to_scipy_sparse_matrix(G)
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)

    labels = [0 for ii in range(100)]+ [1 for ii in range(100)]+ [2 for ii in range(100)]
    num_clusters = 3

    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    
    model = node2vec.fit(window=10, min_count=1, batch_words=4) 
    node_embeddings =  model.wv.vectors


    kmeans = KMeans(n_clusters=num_clusters, init = 'k-means++',n_init = 200, max_iter = 500).fit(node_embeddings)


    ami = adjusted_mutual_info_score( labels, kmeans.labels_,average_method = 'arithmetic')
    ari = adjusted_rand_score( labels , kmeans.labels_)
    list_ami.append(ami)
    list_ari.append(ari)

print('ami : {} +- {}'.format(np.mean(list_ami) , np.std(list_ami)) )
print('ari : {} +- {}'.format(np.mean(list_ari) , np.std(list_ari)) )
