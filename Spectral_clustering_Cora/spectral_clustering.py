import argparse
import numpy as np
import torch
import scipy.sparse as sp
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import networkx as nx
from utils import load_data
from numpy import dot
from torch_geometric.utils import degree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


parser = argparse.ArgumentParser()
parser.add_argument('--centrality', type=str, default='DEGREE', choices=['DEGREE',"KCORE" , "PAGERANK" , "PATHS"])
parser.add_argument('--datadir', type=str, default='./data/',  help='Directory of datasets; default is ./data/')
parser.add_argument('--e1', type=float, default=-1.5)
parser.add_argument('--e2', type=float, default=-1.5)
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--device', type=int, default=0,help='Set CUDA device number; if set to -1, disables cuda.')    
args = parser.parse_args()
# Device
device = torch.device('cuda:'+str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

# Loading the data
adj, features, labels, idx_train, idx_val, idx_test = load_data(path = args.datadir, dataset_name = args.dataset,device =  device)
n = features.size(0)

# Taking only the largest connected componenet
G = nx.from_scipy_sparse_matrix(adj )
largest_cc = list(max(nx.connected_components(G), key=len))
adj = adj.tocsr()[largest_cc,:][:,largest_cc]
labels = labels[largest_cc]
idx_train = idx_train[largest_cc]
idx_val = idx_val[largest_cc]
idx_test = idx_test[largest_cc]
features = features[largest_cc]
n = features.size(0)
adj = sp.csr_matrix(adj.todense())
edge_index, edge_weight = from_scipy_sparse_matrix(adj)
edge_index = edge_index.to(features.device)
G = nx.from_scipy_sparse_matrix(adj )

list_ami = []
list_ari = []
for ii_ in range(5) :
    # Compute the GSO: D^{e1}AD^{e2}

    if args.centrality == 'DEGREE' : 
        row, col = edge_index
        deg = degree(col, n, dtype=float)
        deg_1 = torch.clone(deg)
        deg_2 = torch.clone(deg)

        deg_1 = deg_1.pow(args.e1)
        deg_1[deg_1 == float('inf')] = 0

        deg_2 = deg_2.pow(args.e2)
        deg_2[deg_2 == float('inf')] = 0
        norm = deg_1[row] * deg_2[col]
        edge_index = edge_index
    elif args.centrality == 'PAGERANK' :
        
        row, col = edge_index 
        G = nx.from_scipy_sparse_matrix(adj)
        pr_scores = nx.pagerank(G)
        pr_scores = torch.tensor([pr_scores[k] for k in range(n)]).to(device)
        pr_scores = 1 - pr_scores
        pr_scores_1 = torch.clone(pr_scores)
        pr_scores_2 = torch.clone(pr_scores)
        pr_scores_1 = pr_scores_1.pow( - args.e1)
        pr_scores_1[pr_scores_1 == float('inf')] = 0
        pr_scores_2 = pr_scores_2.pow(- args.e2)
        pr_scores_2[pr_scores_2 == float('inf')] = 0
        
        norm = pr_scores_1[row] * pr_scores_2[col]
        edge_index = edge_index


    elif args.centrality == 'KCORE' :
        row, col = edge_index
        kcores_scores = nx.core_number(G)
        kcores_scores = torch.tensor([kcores_scores[k] for k in range(n)]).to(device)
         
        kcores_scores_1 =  torch.clone(kcores_scores)
        kcores_scores_2 =  torch.clone(kcores_scores)
        
        kcores_scores_1 = kcores_scores_1.pow(args.e1)
        kcores_scores_1[kcores_scores_1 == float('inf')] = 0

        kcores_scores_2 = kcores_scores_2.pow(args.e2)
        kcores_scores_2[kcores_scores_2 == float('inf')] = 0

        norm = kcores_scores_1[row] * kcores_scores_2[col]
        edge_index = edge_index

    elif args.centrality == 'PATHS' :
        row, col = edge_index
        for k_ in range(2) :
            if k_ == 0 :
                n_paths = adj
            else :
                n_paths = dot(n_paths, adj)
        n_paths = torch.tensor( dot(n_paths, sp.csr_matrix(np.ones((n,1)))).todense()).squeeze(1).to(device)

        n_paths_1 = torch.clone(n_paths)
        n_paths_2 = torch.clone(n_paths)

        n_paths_1  = n_paths_1.pow(args.e1)
        n_paths_1[n_paths_1 == float('inf')] = 0
        
        n_paths_2  = n_paths_2.pow(args.e2)
        n_paths_2[n_paths_2 == float('inf')] = 0

        norm = n_paths_1[row] * n_paths_2[col]
        edge_index = edge_index


    # Spliting the labels using the centralities
    try :
        kcores_scores = torch.load( "./Spectral_clustering_Cora/larget_cc_Cora_Centralities/kcore/largest_cc_{}.pth".format(args.dataset)).to(features.device)
    except :
        kcores_scores = nx.core_number(G)
        kcores_scores = torch.tensor([kcores_scores[k] for k in range(n)]).to(device)
        torch.save(kcores_scores,  "./Spectral_clustering_Cora/larget_cc_Cora_Centralities/kcore/largest_cc_{}.pth".format(args.dataset))
    kcores_list = kcores_scores.detach().cpu().numpy().tolist()
    labels = kcores_list
    num_clusters = len(set(labels))

    # Spectral Clustering
    gso_sparse = to_scipy_sparse_matrix(edge_index, norm)
    final_emb = gso_sparse.todense()

    # Directly computing the embedding using eigenvectors
    # The eigenvalues in ascending order
    w, v = np.linalg.eigh(final_emb)
    dim_ = gso_sparse.shape[0]
    reverse_order = np.arange(dim_-1,-1,-1)
    w = w[reverse_order]
    v = v[:,reverse_order]
    v = v[:,:num_clusters]
    kmeans = KMeans(n_clusters=num_clusters, init = 'k-means++',n_init = 200, max_iter = 500).fit(v)


    ami = adjusted_mutual_info_score( labels, kmeans.labels_,average_method = 'arithmetic')
    ari = adjusted_rand_score( labels , kmeans.labels_)
    list_ami.append(ami)
    list_ari.append(ari)

print('ami : {}'.format(list_ami) )
print('ari : {}'.format(list_ari) )

print('ami : {} +- {}'.format(np.mean(list_ami) , np.std(list_ami)) )
print('ari : {} +- {}'.format(np.mean(list_ari) , np.std(list_ari)) )

