################################
# Convolutional models
################################
import torch.nn as nn
import torch.nn.functional as F
from layers import  GCNConv
import torch



class GCN_node_classification(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, init):
        super(GCN_node_classification, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList(
            [GCNConv(input_dim, hidden_dim)] +
            [GCNConv(hidden_dim, hidden_dim) for _ in range(1, num_layers-1)] + 
            [GCNConv(hidden_dim, output_dim)]
        )

        self.dropout = dropout
        if init == 'ADJ' :
            for param in ['m1', 'm2', "m3", 'e1', 'e2','e3','a']:
                if param == 'm2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if init == 'UNORMLAPLACIAN' :
            for param in ['m1', 'm2', "m3", 'e1', 'e2','e3','a']:
                if param == 'm1' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'm2' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                elif param == 'e1'  : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if init == 'SINGLESSLAPLACIAN' :
            for param in ['m1', 'm2', "m3", 'e1', 'e2','e3','a']:
                if param == 'm1' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'm2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e1'  : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if init == 'RWLAPLACIAN' :
            for param in ['m1', 'm2', "m3", 'e1', 'e2','e3','a']:
                if param == 'm2' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                elif param == 'm3' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2'  : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if init == 'SYMLAPLACIAN' :
            for param in ['m1', 'm2', "m3", 'e1', 'e2','e3','a']:
                if param == 'm2' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                elif param == 'm3' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                elif param == 'e3'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))

        if init == 'NORMADJ' :
            for param in ['m1', 'm2', "m3", 'e1', 'e2','e3','a']:
                if param == 'm2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                elif param == 'e3'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                elif param == 'a'  : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))

        if init == 'MEANAGG' :
            for param in ['m1', 'm2', "m3", 'e1', 'e2','e3','a']:
                if param == 'm2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))

    def compute_gso_1(self,row, col, diags):

       # Compute the term m2* (diag**e2)A(diag**e3)
        diags_pow_e2 = diags.pow(self.e2)
        diags_pow_e3 = diags.pow(self.e3)
        
        norm_normalization = diags_pow_e2[row] * diags_pow_e3[col]
        # norm_normalization = norm_normalization * (1-norm_identity)
        gso_1 = self.m2 * norm_normalization 

        return gso_1
    
    def compute_gso_2(self,row_id, col_id, diags, is_null_centrality_mask):
        # Compute the term m1* (diag**e1)
        #norm_1 = norm_diag.pow(self.e1)
        diags_pow_e1 = diags.pow(self.e1)
        norm_1 = diags_pow_e1[row_id]
        norm_1 = self.m1 * norm_1 * (1-is_null_centrality_mask)

        # Compute the term:  m2*a* (diag**e2)I(diag**e3)
        diags_pow_e2 = diags.pow(self.e2)
        diags_pow_e3 = diags.pow(self.e3)
        norm_normalization = diags_pow_e2[row_id] * diags_pow_e3[col_id]
        norm_2_l = self.m2 * self.a * norm_normalization  * (1-is_null_centrality_mask)

        # Compute the first term m3*I
        norm_3 = self.m3

        # Final Norm
        gso_2 = norm_1 + norm_2_l + norm_3
        return gso_2
    
    def forward(self, x, edge_index , edge_index_id, diags, is_null_centrality_mask):
        row, col = edge_index
        row_id, col_id = edge_index_id
        gso_1 = self.compute_gso_1(row, col, diags)
        gso_2 = self.compute_gso_2(row_id, col_id, diags, is_null_centrality_mask)
        h = x
        for i, layer in enumerate(self.conv_layers):
            h1 = layer(h, edge_index , gso_1)
            h2 = layer(h, edge_index_id , gso_2)
            h = h1 + h2
            if i < len(self.conv_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, self.dropout, training=self.training)

        return F.log_softmax(h, dim=1)

