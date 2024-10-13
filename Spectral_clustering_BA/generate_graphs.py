from torch_geometric.utils.random import barabasi_albert_graph
from torch_geometric.utils import degree, stochastic_blockmodel_graph
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import networkx as nx
from scipy.sparse import block_diag
from numpy import dot
import numpy as np
import pickle
import os
import argparse
import glob

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--num_synt_graphs', type=int, default=200)
parser.add_argument('--path', type=str, default='./Spectral_clustering_BA/synthetic_graphs/',  help='Directory of datasets; default is ./data/')
args = parser.parse_args()

# Delate existing files
files = glob.glob(args.path +"*")
for f in files:
    os.remove(f)


# Create new Graphs and Save them
    
for ii_ in range(args.num_synt_graphs) :
        
    # Seperate graphs
    num_nodes = 100 
    list_num_edges = [5,10,15]
    num_blocks = len(list_num_edges)
    dict_degree = {}
    dict_pr_scores = {}
    dict_kcores_scores = {}
    dict_n_paths = {}
    for i_ in range(num_blocks) :
        num_edges = list_num_edges[i_]
        edge_index = barabasi_albert_graph(num_nodes, num_edges)
        adj = to_scipy_sparse_matrix(edge_index)
        if i_ == 0 :
            adj_1 =  adj
        else : 
            adj_1 = block_diag([adj_1,adj ])

    # Aggregate graphs
            
    edge_prob = 0.1
    graph_2 = stochastic_blockmodel_graph([num_nodes for i in range(num_blocks)] , edge_probs =edge_prob * (1 - np.eye(num_blocks)) , directed=False )
    adj_2 = to_scipy_sparse_matrix(graph_2, num_nodes = num_blocks * num_nodes)
    final_adj = adj_1 + adj_2
    G = nx.from_scipy_sparse_matrix(final_adj)

    pickle.dump(G, open(args.path+ 'G_{}.pickle'.format(ii_), 'wb'))