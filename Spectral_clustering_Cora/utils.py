import numpy as np
import scipy.sparse as sp
import torch
import os
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid, TUDataset, Coauthor,  Amazon
from torch_geometric.utils import to_dense_adj, to_undirected
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils.convert import to_scipy_sparse_matrix   
from ogb.nodeproppred import PygNodePropPredDataset
import scipy.sparse as sp

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def fetch_normalization(type):
   switcher = {
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func


def encode_onehot(labels):
    classes = set(labels)
    # print(f'labels: {labels}')
    # print(f'classes: {classes}')
    classes_dict = {c.item(): np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    # print(f'classes_dict: {classes_dict}')
    # print(list(map(classes_dict.get, labels)))
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)


    return labels_onehot

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']


def load_data_gs(path="./data/", dataset="MUTAG", device=None):
    print('Loading {} dataset...'.format(dataset))
    pre_transform = NormalizeFeatures()
    data = TUDataset(path, dataset, pre_transform=pre_transform)[0].to(device)
    print(data)
    features, labels, edges, batch = data.x, data.y, data.edge_index, data.batch
    # adj contains all information from all graphs    
    adj = to_dense_adj(to_undirected(edges)).squeeze()

    return adj, features, labels, batch


def split(dataset, split_type="random", num_train_per_class=20, num_val=500, num_test=1000):
    data = dataset.get(0)
    if split_type=="public" and hasattr(data, "train_mask"):
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    else:
        train_mask = torch.zeros_like(data.y, dtype=torch.bool)
        val_mask = torch.zeros_like(data.y, dtype=torch.bool)
        test_mask = torch.zeros_like(data.y, dtype=torch.bool)

        for c in range(dataset.num_classes):
            idx = (data.y == c).nonzero(as_tuple=False).view(-1)
            idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
            train_mask[idx] = True

        remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]

        val_mask[remaining[:num_val]] = True
        test_mask[remaining[num_val:num_val + num_test]] = True
    return (train_mask, val_mask, test_mask)

def split_random( n, n_train, n_val):
    rnd = np.random.permutation(n)

    train_idx = np.sort(rnd[:n_train])
    val_idx = np.sort(rnd[n_train:n_train + n_val])

    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_val_idx))

    return train_idx, val_idx, test_idx


def load_data(path="./data/", dataset_name="Cora",training_id = 0 , nb_nodes=20, nb_graphs=20, p =None, q=None, device=None):
    print('Loading {} dataset...'.format(dataset_name))

    pre_transform = NormalizeFeatures()



    if dataset_name in {"Cora", "CiteSeer", "PubMed"}:
        data = Planetoid(path, dataset_name, pre_transform=pre_transform)[0].to(device)
        features, labels, edges = data.x, data.y, data.edge_index
        adj = to_scipy_sparse_matrix(to_undirected(edges))
        #adj = to_dense_adj(to_undirected(edges)).squeeze()
        idx_train = data.train_mask
        idx_val = data.val_mask
        idx_test = data.test_mask


        #adj = to_dense_adj(to_undirected(edges)).squeeze()
        idx_train = data.train_mask
        idx_val = data.val_mask
        idx_test = data.test_mask

    elif dataset_name == 'CS' or  dataset_name == 'Physics':
        dataset = Coauthor(root=path, name=dataset_name, transform=pre_transform)
        data = dataset[0].to(device)
        features, labels, edges = data.x, data.y, data.edge_index
        adj = to_scipy_sparse_matrix(to_undirected(edges))
        #adj = to_dense_adj(to_undirected(edges)).squeeze()
        idx_train, idx_val, idx_test = split(dataset, split_type="random", num_train_per_class=20, num_val=500, num_test=1000)

    elif dataset_name == 'Computers' or  dataset_name == 'Photo':
        dataset = Amazon(root=path, name=dataset_name, transform=pre_transform)
        data = dataset[0].to(device)
        features, labels, edges = data.x, data.y, data.edge_index
        adj = to_scipy_sparse_matrix(to_undirected(edges))
        #adj = to_dense_adj(to_undirected(edges)).squeeze()
        idx_train, idx_val, idx_test = split(dataset, split_type="random", num_train_per_class=20, num_val=500, num_test=1000)

    elif dataset_name == 'ogbn-arxiv' :
        # The graph is very large, we cannot work with Dense adjacency matrix
        dataset = PygNodePropPredDataset(name = dataset_name)
        data =  dataset[0].to(device)
        features, labels, edges = data.x, data.y.squeeze(1), data.edge_index
        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
        a_train = torch.zeros(data.num_nodes, dtype=torch.bool)
        a_train[idx_train] = True
        a_val = torch.zeros(data.num_nodes, dtype=torch.bool)
        a_val[idx_val] = True
        a_test = torch.zeros(data.num_nodes, dtype=torch.bool)
        a_test[idx_test] = True
        idx_train, idx_val, idx_test = a_train , a_val, a_test
        adj = to_scipy_sparse_matrix(to_undirected(edges))
        #adj = to_dense_adj(to_undirected(edges)).squeeze()

    else:
        print("Not a correct dataset name!")
        exit()

    return adj, features, labels, idx_train, idx_val, idx_test








def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to_dense()

