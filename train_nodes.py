import time
import argparse
import numpy as np
import os.path as osp
from torch_geometric.utils import to_undirected
import torch
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import scipy.sparse as sp
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.utils import degree, contains_self_loops,remove_self_loops
import torch.optim as optim
from utils import load_data, accuracy
from models.gcn_conv import GCN_node_classification
from datetime import datetime
from pathlib import Path
import networkx as nx
from numpy import dot
  

########################################################################################
# Train and test functions 
########################################################################################

def train(epoch):
    t = time.time()
    model.train()
    
    optimizer.zero_grad()
    output = model(features.float(), edge_index.detach() ,edge_index_id.detach(), diags.float(), is_null_centrality_mask ) 
    
    loss_train = criterion(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    acc_train = accuracy(output[idx_train], labels[idx_train])
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features.float(), edge_index , edge_index_id.detach(),diags.float(), is_null_centrality_mask)

    loss_val = criterion(output[idx_val], labels[idx_val]).detach()
    acc_val = accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t)
        #   'cond: {:.1f}'.format(condition_number(model.gen_adj))
          )

    whole_state = {
        'epoch': epoch,
        'model_state_dict': {key:val.clone() for key,val in model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_train': loss_train.detach(),
        'loss_val': loss_val.detach(),
        'acc_train': acc_train.detach(),
        'acc_val': acc_val.detach(),
        }
    return whole_state


def test():
    model.eval()
    output = model(features.float(), edge_index ,edge_index_id.detach(), diags.float(), is_null_centrality_mask)
    loss_test = criterion(output[idx_test], labels[idx_test])
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()


########################################################################################
# Parse arguments 
########################################################################################

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--centrality', type=str, default='DEGREE',  choices=['DEGREE', 'KCORE','PAGERANK','PATHS'])
parser.add_argument('--datadir', type=str, default='./data/',  help='Directory of datasets; default is ./data/')
parser.add_argument('--init', type=str, default='MEANAGG',  choices=['ADJ', 'UNORMLAPLACIAN','SINGLESSLAPLACIAN','RWLAPLACIAN','SYMLAPLACIAN','NORMADJ', 'MEANAGG'])
parser.add_argument('--dataset', type=str, default="Cora", help='Dataset name; default is Cora' ,choices=["Cora" ,"CiteSeer", "PubMed", "CS", "ogbn-arxiv", 'Computers','Photo', "Physics"])
parser.add_argument('--device', type=int, default=0,help='Set CUDA device number; if set to -1, disables cuda.')    
parser.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--exp_lr', type=float, default=0.005,help='Initial learning rate for exponential parameters.')
parser.add_argument('--lr_patience', type=float, default=50, help='Number of epochs waiting for the next lr decay.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,help='Number of hidden units.')
parser.add_argument('--num_layers', type=int, default=2,  help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')
args = parser.parse_args()

if args.dataset == 'ogbn-arxiv' :
    args.hidden = 512
if args.dataset == 'Cora' :
    args.lr = 0.01
    args.hidden = 64
    args.dropout = 0.8

elif args.dataset == 'CiteSeer' :
    args.lr = 0.01
    args.hidden = 64
    args.dropout = 0.4
elif args.dataset == 'PubMed' :
    args.lr = 0.01
    args.hidden = 64
    args.dropout = 0.2
elif args.dataset == 'CS' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.4
elif args.dataset == 'genius' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.8
elif args.dataset == 'Penn94' :
    args.lr = 0.01
    args.hidden = 64
    args.dropout = 0.2
    
elif args.dataset == 'Computers' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.2

elif args.dataset == 'Photo' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.6





device = torch.device('cuda:'+str(args.device)) if torch.cuda.is_available() else torch.device('cpu')


##################################################################m######################
# Data loading and model setup 
#########################################################################################


## List to store the learned hyperparameters
all_test_acc = []
all_e1 = []
all_e2 = []
all_e3 = []
all_m1 = []
all_m2 = []
all_m3 = []
all_a = []


for t_ in range(10) :
    # We split the graph into train/val/test inside the loops as the split for some datasets is random
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path = args.datadir, dataset_name = args.dataset,training_id = t_,device =  device )
    n = features.size(0)

    edge_index_without_loops, edge_weight_without_loops = from_scipy_sparse_matrix(adj)
    edge_index_without_loops = edge_index_without_loops.to(features.device)

    if args.centrality == 'DEGREE' :
        row, col = edge_index_without_loops
        deg = degree(col, n, dtype=features.dtype)
        diags = deg

    elif args.centrality == 'PAGERANK' :
        
        row, col = edge_index_without_loops
        try : 
            pr_scores = torch.load( "./pagerank/{}.pth".format(args.dataset)).to(features.device)
            print("PAGERANK Loaded ")
        except : 
            # Remove self loops to compute K cores
            G = nx.from_scipy_sparse_matrix(adj)
            pr_scores = nx.pagerank(G)
            pr_scores = torch.tensor([pr_scores[k] for k in range(n)]).to(features.device)
            torch.save(pr_scores, "./centralities/pagerank/{}.pth".format(args.dataset))

        pr_scores = 1 - pr_scores
        diags = pr_scores


    elif args.centrality == 'KCORE' :
        any_self_loops = contains_self_loops(edge_index_without_loops)
        if  any_self_loops :
            new_edge_index_without_loops = remove_self_loops(edge_index_without_loops)[0]
            row, col = new_edge_index_without_loops
            adj_remove_self_loops = to_scipy_sparse_matrix(to_undirected(new_edge_index_without_loops))
        else:
            row, col = edge_index_without_loops
        
        try : 
            kcores_scores = torch.load( "./centralities/kcore/{}.pth".format(args.dataset)).to(features.device)
            print("Kcora Loaded ")
        except : 
            # Remove self loops to compute K cores
            if any_self_loops:
                G = nx.from_scipy_sparse_matrix(adj_remove_self_loops )
            else :
                G = nx.from_scipy_sparse_matrix(adj )
            kcores_scores = nx.core_number(G)
            kcores_scores = torch.tensor([kcores_scores[k] for k in range(n)]).to(features.device)
            torch.save(kcores_scores, "./centralities/kcore/{}.pth".format(args.dataset))

        diags = kcores_scores

    elif args.centrality == 'PATHS' :
        row, col = edge_index_without_loops
        try : 
            n_paths = torch.load( "./centralities/paths/{}.pth".format(args.dataset)).to(features.device)
            print("paths Loaded ")
        except :
            for k_ in range(args.num_layers) :
                if k_ == 0 :
                    n_paths = adj
                else :
                    n_paths = dot(n_paths, adj)
            n_paths = torch.tensor( dot(n_paths, sp.csr_matrix(np.ones((n,1)))).todense()).squeeze(1).to(features.device)
            torch.save(n_paths, "./centralities/paths/{}.pth".format(args.dataset))
        diags = n_paths




    # Now that we have computed the centralities without self loops, we add self loops in the edge index
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    edge_index = edge_index.to(features.device)
    row, col = edge_index
    diags = diags.clone()
    edge_index_id, edge_weight_id = from_scipy_sparse_matrix(sp.identity(n))
    edge_index_id = edge_index_id .to(features.device)
   
    indices_with_zeros_d = (diags == 0).nonzero(as_tuple=True)[0].cpu().numpy().tolist()
    diags = diags * (1-(diags==0)*1) + (diags==0)*1
    is_null_centrality_mask = torch.tensor([1 if (edge_index_id[0,i].item() in indices_with_zeros_d or edge_index_id[0,i].item() in indices_with_zeros_d) else 0 for i in range(edge_index_id.size(-1))  ]).to(diags.device)
    

    #####################
    ####  Training 
    ##########
    # Model and optimizer
    model = GCN_node_classification(input_dim=features.shape[1],
                hidden_dim=args.hidden,
                output_dim=labels.max().item() + 1,
                num_layers=args.num_layers,
                dropout=args.dropout, init = args.init).to(device)

    # Exponential parameters have a different learning rate than other multiplicative parameters
    exp_param_list = ['e1', 'e2' , 'e3']
    exp_params = list(filter(lambda kv: kv[0] in exp_param_list, model.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in exp_param_list, model.named_parameters()))
    exp_params = [param[1] for param in exp_params]
    base_params = [param[1] for param in base_params]
    
    optimizer = optim.Adam([
                            {'params': base_params, 'lr':args.lr},
                            {'params': exp_params, 'lr': args.exp_lr}
                            ], lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    lr_scheduler = None
    if args.lr_patience > 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_patience, gamma=0.6)

    # Train model
    t_total = time.time() 
    m1_values, m2_values, e1_values, e2_values = [], [], [], []

    states = []
    for epoch in range(1,args.epochs+1):
        state = train(epoch)
        states.append(state)
        if args.lr_patience > 0:
            lr_scheduler.step()

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    acc_test =  test()
    all_test_acc.append(acc_test)
    all_e1.append(model.e1.item())
    all_e2.append(model.e2.item())
    all_e3.append(model.e3.item())
    all_m1.append(model.m1.item())
    all_m2.append(model.m2.item())
    all_m3.append(model.m3.item())
    all_a.append(model.a.item())

print("mean_accuracy : ", np.mean(all_test_acc)  ," , std_accuracy : " , np.std(all_test_acc) )




