import torch
import torch.nn as nn

# predict flow helpers
def deltaphi(phi1, phi2):
    return torch.fmod(phi1 - phi2 + math.pi, 2*math.pi) - math.pi

def deltaR(p1, p2):
    deta = p1[:,1]-p2[:,1]
    dphi = deltaphi(p1[:,2], p2[:,2])
    return torch.sqrt(torch.square(deta)  + torch.square(dphi))

def get_emd(x, edge_index, fij, u, batch):
    R = 0.4
    row, col = edge_index
    dR = deltaR(x[row], x[col])
    edge_batch = batch[row]
    emd = scatter_add(fij*dR/R, edge_batch, dim=0) + torch.abs(u[:,0]-u[:,1])
    return emd

class LossFunction:
    def __init__(self, lossname, lam1=1, lam2=100):
        if lossname == 'mse':
            loss = nn.MSELoss(reduction='mean')
        else:
            loss = getattr(self, lossname)
        self.loss_ftn = loss
        self.name(lossname)
        self.lam1 = lam1
        self.lam2 = lam2

    def symm_loss_1(self, preds, ys, emd_1, emd_2):
        mse = nn.MSELoss(reduction='mean')
        batch_loss = mse(ys, preds) + mse(emd_1, emd_2)
        return batch_loss

    def symm_loss_2(self, preds, ys):
        mse = nn.MSELoss(reduction='mean')
        batch_loss = mse(ys, preds) + self.lam1 * torch.mean(torch.pow(preds, 2))
        return batch_loss
    
    def predict_flow(self, data, preds):
        loss1 = mse(get_emd(data.x, data.edge_index, preds.squeeze(), data.u, data.batch).unsqueeze(-1), data.y)
        loss2 = mse(preds, data.edge_y)
        batch_loss = lam1*loss1 + lam2*loss2
        return batch_loss