import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""

class GatedTestLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
        
        self.P = nn.Parameter(torch.rand(output_dim)*1e-3+1) 
    
    def pNorm(self, nodes):
        P = torch.clamp(self.P,1,100)
        """ fn.u_add_v(nodes['Dh'], nodes['Eh'], nodes['DEh']) 
        nodes['e'] = nodes['DEh'] + nodes['Ce']
        nodes['sigma'] = torch.sigmoid(nodes['e'])
        fn.u_mul_e(nodes['Bh'],nodes['sigma'], nodes['test'])
        fn.sum(nodes['test'], nodes['sum_sigma_h'])
        fn.copy_e(nodes['sigma'], nodes['test2'])
        fn.sum(nodes['test2'], nodes['sum_sigma'])
        nodes['h'] = nodes['Ah'] + nodes['sum_sigma_h'] / (nodes['sum_sigma'] + 1e-6)  """
        #h = (F.relu(nodes.mailbox['m'])).pow(P)
        print(nodes.ndata['h'])
        h = torch.abs(nodes.mailbox['m']).pow(P)
        return {'neigh': torch.sum(h, dim=1).pow(1/P)}

    def pNorm_edges(self, edges):
        """ fn.u_add_v(nodes['Dh'], nodes['Eh'], nodes['DEh']) 
        nodes['e'] = nodes['DEh'] + nodes['Ce']
        nodes['sigma'] = torch.sigmoid(nodes['e'])
        fn.u_mul_e(nodes['Bh'],nodes['sigma'], nodes['test'])
        fn.sum(nodes['test'], nodes['sum_sigma_h'])
        fn.copy_e(nodes['sigma'], nodes['test2'])
        fn.sum(nodes['test2'], nodes['sum_sigma'])
        nodes['h'] = nodes['Ah'] + nodes['sum_sigma_h'] / (nodes['sum_sigma'] + 1e-6)  """
        #h = (F.relu(nodes.mailbox['m'])).pow(P)
        h = torch.abs(nodes.mailbox['m']).pow(P)
        return {'neigh': torch.sum(h, dim=1).pow(1/P)}

    def updata_all_example(self, graph):
        # store the result in graph.ndata['ft']g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh')) # . u_add_v = computes message on edge by elementwise addition. 
                                                     #   "DEh" output field. Apply edges - update edge features. Why edge features? 

        p = torch.clamp(self.P,1,100)
        graph.edata['e'] = graph.edata['DEh'] + graph.edata['Ce'] #  Bh + Ah. ????

        graph.edata['sigma'] = torch.sigmoid(graph.edata['e']) # n_{ij}

        graph.ndata['Bh_pow'] = torch.abs(graph.ndata['Bh']).pow(p)
        graph.edata['sig_pow'] = torch.abs(graph.edata['sigma']).pow(p)

        graph.update_all(fn.u_mul_e('Bh_pow', 'sig_pow', 'm'), fn.sum('m', 'sum_sigma_h')) # u_mul_e = elementwise mul. Output "m" = n_{ij}***Vh. Then sum! 
                                                                                 # Update_all - send messages through all edges and update all nodes.
        

        graph.update_all(fn.copy_e('sig_pow', 'm'), fn.sum('m', 'sum_sigma')) # copy_e - eqv to 'm': graph.edata['sigma']. Output "m". Then sum. 
                                                                        # Again, send messages and update all nodes. Why do this step?????
        
        graph.ndata['h'] = graph.ndata['Ah'] + (graph.ndata['sum_sigma_h'] / (graph.ndata['sum_sigma'] + 1e-6)).pow(torch.div(1,p)) # Uh + sum()

        #graph.update_all(self.message_func,self.reduce_func) 
        h = graph.ndata['h'] # result of graph convolution
        e = graph.edata['e'] # result of graph convolution
        # Call update function outside of update_all
        final_ft = graph.ndata['ft'] * 2
        return h, e

    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        e_in = e # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 

        h, e = g.updata_all_example(self, g)
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
        
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)

    
##############################################################
#
# Additional layers for edge feature/representation analysis
#
##############################################################


class GatedGCNLayerEdgeFeatOnly(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)

    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        #g.update_all(self.message_func,self.reduce_func) 
        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'e'))
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
        h = g.ndata['h'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization    
        
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)


##############################################################


class GatedGCNLayerIsotropic(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)

    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h)
        #g.update_all(self.message_func,self.reduce_func) 
        g.update_all(fn.copy_u('Bh', 'm'), fn.sum('m', 'sum_h'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_h']
        h = g.ndata['h'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization    
        
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)
    
