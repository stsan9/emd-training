import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import DynamicEdgeConv, EdgeConv, global_mean_pool
from torch_scatter import scatter_mean, scatter_add
import math

class EdgeNet(nn.Module):
    def __init__(self, input_dim=3, big_dim=32, bigger_dim=128, global_dim=2, output_dim=1, aggr='mean'):
        super(EdgeNet, self).__init__()
        convnn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
        )
                
        self.batchnorm = nn.BatchNorm1d(input_dim)
        
        self.batchnormglobal = nn.BatchNorm1d(global_dim)        
        
        self.outnn = nn.Sequential(nn.Linear(big_dim+global_dim, bigger_dim),
                                   nn.ReLU(),
                                   nn.Linear(bigger_dim, bigger_dim),
                                   nn.ReLU(),
                                   nn.Linear(bigger_dim, output_dim)
        )
        
        self.conv = EdgeConv(nn=convnn,aggr=aggr)

    def forward(self, data):
        data.x = self.batchnorm(data.x)
        data.x = self.conv(data.x,data.edge_index)
        u1 = self.batchnormglobal(data.u)
        u2 = scatter_mean(data.x, data.batch, dim=0)
        data.u = torch.cat([u1, u2],dim=-1)
        return self.outnn(data.u)    
        
class DynamicEdgeNet(nn.Module):
    def __init__(self, input_dim=3, big_dim=128, bigger_dim=256, global_dim=2, output_dim=1, k=16, aggr='mean'):
        super(DynamicEdgeNet, self).__init__()
        convnn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
        )
                
        self.batchnorm = nn.BatchNorm1d(input_dim)
        
        self.batchnormglobal = nn.BatchNorm1d(global_dim)        
        
        self.outnn = nn.Sequential(nn.Linear(big_dim+global_dim, bigger_dim),
                                   nn.ReLU(),
                                   nn.Linear(bigger_dim, bigger_dim),
                                   nn.ReLU(),
                                   nn.Linear(bigger_dim, output_dim)
        )
        
        self.conv = DynamicEdgeConv(nn=convnn,aggr=aggr, k=k)

    def forward(self, data):
        data.x = self.batchnorm(data.x)
        data.x = self.conv(data.x, data.batch)
        u1 = self.batchnormglobal(data.u)
        u2 = scatter_mean(data.x, data.batch, dim=0)
        data.u = torch.cat([u1, u2],dim=-1)
        return self.outnn(data.u)
    
class DeeperDynamicEdgeNet(nn.Module):
    def __init__(self, input_dim=3, big_dim=32, bigger_dim=256, global_dim=2, output_dim=1, k=16, aggr='mean'):
        super(DeeperDynamicEdgeNet, self).__init__()
        convnn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.BatchNorm1d(big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.BatchNorm1d(big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.BatchNorm1d(big_dim),
                               nn.ReLU(),
        )
        convnn2 = nn.Sequential(nn.Linear(2*(big_dim+input_dim), big_dim*2),
                               nn.BatchNorm1d(big_dim*2),
                               nn.ReLU(),
                               nn.Linear(big_dim*2, big_dim*2),
                               nn.BatchNorm1d(big_dim*2),
                               nn.ReLU(),
                               nn.Linear(big_dim*2, big_dim*2),
                               nn.BatchNorm1d(big_dim*2),
                               nn.ReLU(),
        )
        convnn3 = nn.Sequential(nn.Linear(2*(big_dim*2+input_dim), big_dim*4),
                               nn.BatchNorm1d(big_dim*4),
                               nn.ReLU(),
                               nn.Linear(big_dim*4, big_dim*4),
                               nn.BatchNorm1d(big_dim*4),
                               nn.ReLU(),
                               nn.Linear(big_dim*4, big_dim*4),
                               nn.BatchNorm1d(big_dim*4),
                               nn.ReLU(),
        )
                
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.batchnormglobal = nn.BatchNorm1d(global_dim)
        self.outnn = nn.Sequential(nn.Linear(big_dim*4+input_dim+global_dim, bigger_dim),
                                   nn.BatchNorm1d(bigger_dim),
                                   nn.ReLU(),
                                   nn.Linear(bigger_dim, output_dim)
        )
        
        self.conv = DynamicEdgeConv(nn=convnn, aggr=aggr, k=k)
        self.conv2 = DynamicEdgeConv(nn=convnn2, aggr=aggr, k=k)
        self.conv3 = DynamicEdgeConv(nn=convnn3, aggr=aggr, k=k)

    def forward(self, data):
        x1 = self.batchnorm(data.x)        
        x2 = self.conv(data.x, data.batch)        
        data.x = torch.cat([x1, x2],dim=-1)
        x2 = self.conv2(data.x, data.batch)          
        data.x = torch.cat([x1, x2],dim=-1)
        x2 = self.conv3(data.x, data.batch)
        data.x = torch.cat([x1, x2],dim=-1)        
        u1 = self.batchnormglobal(data.u)
        u2 = scatter_mean(data.x, data.batch, dim=0)
        data.u = torch.cat([u1, u2],dim=-1)       
        return self.outnn(data.u)


class DeeperDynamicEdgeNetPredictFlow(nn.Module):
    def __init__(self, input_dim=3, big_dim=32, bigger_dim=256, global_dim=2, output_dim=1, k=16, aggr='mean'):
        super(DeeperDynamicEdgeNetPredictFlow, self).__init__()
        convnn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.BatchNorm1d(big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.BatchNorm1d(big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.BatchNorm1d(big_dim),
                               nn.ReLU(),
        )
        convnn2 = nn.Sequential(nn.Linear(2*(big_dim+input_dim), big_dim*2),
                               nn.BatchNorm1d(big_dim*2),
                               nn.ReLU(),
                               nn.Linear(big_dim*2, big_dim*2),
                               nn.BatchNorm1d(big_dim*2),
                               nn.ReLU(),
                               nn.Linear(big_dim*2, big_dim*2),
                               nn.BatchNorm1d(big_dim*2),
                               nn.ReLU(),
        )
        convnn3 = nn.Sequential(nn.Linear(2*(big_dim*2+input_dim), big_dim*4),
                               nn.BatchNorm1d(big_dim*4),
                               nn.ReLU(),
                               nn.Linear(big_dim*4, big_dim*4),
                               nn.BatchNorm1d(big_dim*4),
                               nn.ReLU(),
                               nn.Linear(big_dim*4, big_dim*4),
                               nn.BatchNorm1d(big_dim*4),
                               nn.ReLU(),
        )
                
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.outnn = nn.Sequential(nn.Linear(big_dim*4*2+input_dim*2, bigger_dim),
                                   nn.BatchNorm1d(bigger_dim),
                                   nn.ReLU(),
                                   nn.Linear(bigger_dim, output_dim)
        )
        
        self.conv = DynamicEdgeConv(nn=convnn, aggr=aggr, k=k)
        self.conv2 = DynamicEdgeConv(nn=convnn2, aggr=aggr, k=k)
        self.conv3 = DynamicEdgeConv(nn=convnn3, aggr=aggr, k=k)

    def forward(self, data):
        x1 = self.batchnorm(data.x)        
        x2 = self.conv(data.x, data.batch)        
        data.x = torch.cat([x1, x2],dim=-1)
        x2 = self.conv2(data.x, data.batch)          
        data.x = torch.cat([x1, x2],dim=-1)
        x2 = self.conv3(data.x, data.batch)
        data.x = torch.cat([x1, x2],dim=-1)        
        row,col = data.edge_index        
        return self.outnn(torch.cat([data.x[row],data.x[col]],dim=-1))#.squeeze(-1)



class DeeperDynamicEdgeNetPredictEMDFromFlow(nn.Module):
    def __init__(self, input_dim=3, big_dim=32, bigger_dim=256, global_dim=2, output_dim=1, k=16, aggr='mean'):
        super(DeeperDynamicEdgeNetPredictEMDFromFlow, self).__init__()
        convnn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.BatchNorm1d(big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.BatchNorm1d(big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.BatchNorm1d(big_dim),
                               nn.ReLU(),
        )
        convnn2 = nn.Sequential(nn.Linear(2*(big_dim+input_dim), big_dim*2),
                               nn.BatchNorm1d(big_dim*2),
                               nn.ReLU(),
                               nn.Linear(big_dim*2, big_dim*2),
                               nn.BatchNorm1d(big_dim*2),
                               nn.ReLU(),
                               nn.Linear(big_dim*2, big_dim*2),
                               nn.BatchNorm1d(big_dim*2),
                               nn.ReLU(),
        )
        convnn3 = nn.Sequential(nn.Linear(2*(big_dim*2+input_dim), big_dim*4),
                               nn.BatchNorm1d(big_dim*4),
                               nn.ReLU(),
                               nn.Linear(big_dim*4, big_dim*4),
                               nn.BatchNorm1d(big_dim*4),
                               nn.ReLU(),
                               nn.Linear(big_dim*4, big_dim*4),
                               nn.BatchNorm1d(big_dim*4),
                               nn.ReLU(),
        )
                
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.outnn = nn.Sequential(nn.Linear(big_dim*4*2+input_dim*2, bigger_dim),
                                   nn.BatchNorm1d(bigger_dim),
                                   nn.ReLU(),
                                   nn.Linear(bigger_dim, bigger_dim),
                                   nn.BatchNorm1d(bigger_dim),
                                   nn.ReLU(),
                                   nn.Linear(bigger_dim, output_dim)
        )
        
        self.conv = DynamicEdgeConv(nn=convnn, aggr=aggr, k=k)
        self.conv2 = DynamicEdgeConv(nn=convnn2, aggr=aggr, k=k)
        self.conv3 = DynamicEdgeConv(nn=convnn3, aggr=aggr, k=k)

    def get_emd(self, x, edge_index, fij, u, batch):
        R = 0.4
        row, col = edge_index
        dR = torch.sqrt(torch.square(x[row][:,1] - x[col][:,1])  + torch.square(torch.fmod(x[row][:,2] - x[col][:,2] + math.pi, 2*math.pi) - math.pi))
        emd = scatter_add(dR*fij, batch[row], dim=0) + torch.abs(u[:,0]-u[:,1])
        return emd.unsqueeze(-1)

    def forward(self, data):
        x0 = data.x
        x1 = self.batchnorm(data.x)        
        x2 = self.conv(data.x, data.batch)
        data.x = torch.cat([x1, x2],dim=-1)
        x2 = self.conv2(data.x, data.batch)          
        data.x = torch.cat([x1, x2],dim=-1)
        x2 = self.conv3(data.x, data.batch)
        data.x = torch.cat([x1, x2],dim=-1)        
        row,col = data.edge_index        
        fij = self.outnn(torch.cat([data.x[row],data.x[col]],dim=-1)).squeeze(-1)
        return self.get_emd(x0, data.edge_index, fij, data.u, data.batch)


