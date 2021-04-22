#!/usr/bin/env python
import numpy as np
import math
import matplotlib.pyplot as plt
import energyflow as ef
import torch
import torch.nn as nn
import os.path as osp
import os
import sys
import tqdm
import random
import logging
from torch_scatter import scatter_add
from graph_data import GraphDataset, ONE_HUNDRED_GEV
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from torch.utils.data import random_split

import models
import loss_ftns
from plot import make_plots
from process_util import remove_dupes, pair_dupes

plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'serif'
torch.manual_seed(0)
np.random.seed(0)

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

@torch.no_grad()
def test(model, loader, total, batch_size, predict_flow, lam1, lam2, symm_loss=None, symm_lam=None):
    model.eval()
    
    mse = nn.MSELoss(reduction='mean')

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        data = data.to(device)
        if predict_flow:
            x = data.x
            batch_output = model(data)
            loss1 = mse(get_emd(x, data.edge_index, batch_output.squeeze(), data.u, data.batch).unsqueeze(-1), data.y)
            loss2 = mse(batch_output, data.edge_y)
            batch_loss = lam1*loss1 + lam2*loss2
        elif symm_loss is not None:
            batch_output = model(data)
            if symm_loss == loss_ftns.symm_loss_1:
                # loss = mse(y, pred) + mse(emd1, emd2)
                batch_output, emd_1, emd_2 = batch_output
                batch_loss = symm_loss(batch_output, data.y, emd_1, emd_2)
            if symm_loss == loss_ftns.symm_loss_2:
                # loss = mse(y, pred) + lam * pred^2
                batch_output, _, _ = batch_output
                batch_loss = symm_loss(batch_output, data.y, symm_lam if symm_lam is not None else 0.001)
        else:
            batch_output = model(data)
            batch_loss = mse(batch_output, data.y)
        batch_loss_item = batch_loss.item()
        sum_loss += batch_loss_item
        t.set_description("loss = %.5f" % (batch_loss_item))
        t.refresh() # to show immediately the update

    return sum_loss/(i+1)

def train(model, optimizer, loader, total, batch_size, predict_flow, lam1, lam2, symm_loss=None, symm_lam=None):
    model.train()
    
    mse = nn.MSELoss(reduction='mean')

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        data = data.to(device)
        optimizer.zero_grad()
        if predict_flow:
            x = data.x
            batch_output = model(data)
            loss1 = mse(get_emd(x, data.edge_index, batch_output.squeeze(), data.u, data.batch).unsqueeze(-1), data.y)
            loss2 = mse(batch_output, data.edge_y)
            batch_loss = lam1*loss1 + lam2*loss2
        elif symm_loss is not None:
            batch_output = model(data)
            if symm_loss == loss_ftns.symm_loss_1:
                # loss = mse(y, pred) + mse(emd1, emd2)
                batch_output, emd_1, emd_2 = batch_output
                batch_loss = symm_loss(batch_output, data.y, emd_1, emd_2)
            if symm_loss == loss_ftns.symm_loss_2:
                # loss = mse(y, pred) + lam * pred^2
                batch_output, _, _ = batch_output
                batch_loss = symm_loss(batch_output, data.y, symm_lam if symm_lam is not None else 0.001)
        else:
            batch_output = model(data)
            batch_loss = mse(batch_output, data.y)
        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description("loss = %.5f" % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()
    
    return sum_loss/(i+1)

def plot_losses(losses, val_losses, model_fname, output_dir):
    fig, ax = plt.subplots(figsize =(5, 5)) 
    plt.plot(losses, marker='o',label='Training', alpha=0.5)
    plt.plot(val_losses, marker='o',label = 'Validation', alpha=0.5)
    plt.legend()
    ax.set_ylabel('Loss') 
    ax.set_xlabel('Epoch') 
    fig.savefig(osp.join(output_dir,model_fname+'_loss.pdf'))
    fig.savefig(osp.join(output_dir,model_fname+'_loss.png'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-dir", type=str, help="Output directory for models and plots.", required=False, 
                        default='models2/')
    parser.add_argument("--input-dir", type=str, help="Input directory for datasets.", required=False,
                        default='/energyflowvol/datasets/')
    parser.add_argument("--model", choices=['EdgeNet', 'DynamicEdgeNet','DeeperDynamicEdgeNet','DeeperDynamicEdgeNetPredictFlow',
                                            'DeeperDynamicEdgeNetPredictEMDFromFlow','SymmetricDDEdgeNet'], 
                        help="Model name", required=False, default='DeeperDynamicEdgeNet')
    parser.add_argument("--symm-loss", choices=['symm_loss_1', 'symm_loss_2'], help="special loss; else use standard mse", required=False, default=None)
    parser.add_argument("--symm-lam", type=float, help="lambda term for symm_loss_2", default=None, required=False)
    parser.add_argument("--lhco", action='store_true', help="Using lhco dataset (diff processing)", default=False, required=False)
    parser.add_argument("--n-jets", type=int, help="number of jets", required=False, default=100)
    parser.add_argument("--n-events-merge", type=int, help="number of events to merge", required=False, default=1)
    parser.add_argument("--batch-size", type=int, help="batch size", required=False, default=100)
    parser.add_argument("--n-epochs", type=int, help="number of epochs", required=False, default=100)
    parser.add_argument("--patience", type=int, help="patience for early stopping", required=False, default=10)
    parser.add_argument("--predict-flow", action="store_true", help="predict edge flow instead of emdval", required=False)
    parser.add_argument("--remove-dupes", action="store_true", help="remove data that had the same jet pair in different order (leave one ver)", required=False, default=False)
    parser.add_argument("--pair-dupes", action="store_true", help="pair data that use the same jet pair", required=False, default=False)
    parser.add_argument("--lam1", type=float, help="lambda1 for EMD loss term", default=1, required=False)
    parser.add_argument("--lam2", type=float, help="lambda2 for fij loss term", default=100, required=False)
    args = parser.parse_args()

    # create output directory
    os.makedirs(args.output_dir,exist_ok=True)

    # basic checks
    if args.model == 'SymmetricDDEdgeNet' and args.symm_loss is None:
        exit("Specify args.symm_loss when using symmetric network")
    if (args.symm_loss is not loss_ftns.symm_loss_2) and (args.symm_lam is not None):
        exit("--symm-lam is for use with symm_loss_2")
    if args.remove_dupes and args.pair_dupes:
        exit("can't remove dupes and pair dupes at the same time")

    # log arguments
    logging.basicConfig(filename=osp.join(args.output_dir, "logs.log"), filemode='w', level=logging.DEBUG, format='%(asctime)s | %(levelname)s: %(message)s')
    for arg, value in sorted(vars(args).items()):
            logging.info("Argument %s: %r", arg, value)

    # create model
    model_class = getattr(models, args.model)
    input_dim = 4
    big_dim = 32
    bigger_dim = 128
    global_dim = 2
    output_dim = 1
    batch_size = args.batch_size
    predict_flow = args.predict_flow
    lr = 0.001
    device = 'cuda:0'
    model_fname = args.model
    modpath = osp.join(args.output_dir,model_fname+'.best.pth')
    lam1 = args.lam1
    lam2 = args.lam2
    model = model_class(input_dim=input_dim, big_dim=big_dim, bigger_dim=bigger_dim, 
                        global_dim=global_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    try:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(modpath, map_location=torch.device('cuda')))
        else:
            model.load_state_dict(torch.load(modpath, map_location=torch.device('cpu')))
        logging.debug("Using trained model")
    except:
        logging.debug("Creating a new model")

    # get appropriate loss function if needed
    if args.symm_loss is not None:
        symm_loss = getattr(loss_ftns, args.symm_loss)
    else:
        symm_loss = None

    # load data
    logging.debug("Loading dataset...")
    gdata = GraphDataset(root=args.input_dir, n_jets=args.n_jets, n_events_merge=args.n_events_merge, lhco=args.lhco, lhco_back=args.lhco_back)
    logging.debug("Dataset loaded.")

    # shuffling data and handling pairs
    logging.debug("Preparing data to shuffle...")
    bag = []
    for g in gdata:
        bag += g
    if args.remove_dupes:
        bag = remove_dupes(bag)
    elif args.pair_dupes:
        bag = pair_dupes(bag)
    random.Random(0).shuffle(bag)
    logging.debug("Shuffled.")

    # split dataset
    fulllen = len(bag)
    train_len = int(0.8 * fulllen)
    tv_len = int(0.10 * fulllen)
    train_dataset = bag[:train_len]
    valid_dataset = bag[train_len:train_len + tv_len]
    test_dataset  = bag[train_len + tv_len:]
    if args.pair_dupes:
        train_dataset = sum(train_dataset, [])
        valid_dataset = sum(valid_dataset, [])
        test_dataset  = sum(test_dataset, [])
    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)
    test_samples = len(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    # train loop
    n_epochs = args.n_epochs
    patience = args.patience
    stale_epochs = 0
    best_valid_loss = test(model, valid_loader, valid_samples, batch_size, predict_flow, lam1, lam2, symm_loss, args.symm_lam)
    losses = []
    val_losses = []
    for epoch in range(0, n_epochs):
        loss = train(model, optimizer, train_loader, train_samples, batch_size, predict_flow, lam1, lam2, symm_loss, args.symm_lam)
        losses.append(loss)
        valid_loss = test(model, valid_loader, valid_samples, batch_size, predict_flow, lam1, lam2, symm_loss, args.symm_lam)
        val_losses.append(valid_loss)
        print('Epoch: {:02d}, Training Loss:   {:.4f}'.format(epoch, loss))
        print('               Validation Loss: {:.4f}'.format(valid_loss))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('New best model saved to:',modpath)
            torch.save(model.state_dict(),modpath)
            stale_epochs = 0
        else:
            print('Stale epoch')
            stale_epochs += 1
        if stale_epochs >= patience:
            print('Early stopping after %i stale epochs'%patience)
            break
    else:
        test_dataset  = bag
        test_samples = len(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    # test model
    model.load_state_dict(torch.load(modpath))
    ys = []
    preds = []
    diffs = []
    
    t = tqdm.tqdm(enumerate(test_loader),total=test_samples/batch_size)
    model.eval()
    for i, data in t:
        data.to(device)
        if predict_flow:
            # just to double-check the formula, this gives the same answer as data.y.squeeze()
            #true_emd = get_emd(data.x, data.edge_index, data.edge_y.squeeze(), data.u, data.batch)
            true_emd = data.y.squeeze()
            learn_emd = get_emd(data.x, data.edge_index, model(data).squeeze(), data.u, data.batch)
        else:
            true_emd = data.y
            learn_emd = model(data)
            if args.model == "SymmetricDDEdgeNet":
                learn_emd = learn_emd[0]    # toss unecessary terms

        ys.append(true_emd.cpu().numpy().squeeze()*ONE_HUNDRED_GEV)
        preds.append(learn_emd.cpu().detach().numpy().squeeze()*ONE_HUNDRED_GEV)
    
    ys = np.concatenate(ys)   
    preds = np.concatenate(preds)
    losses = np.array(losses)
    val_losses = np.array(val_losses)
    np.save(osp.join(args.output_dir,model_fname+'_ys.npy'),ys)
    np.save(osp.join(args.output_dir,model_fname+'_preds.npy'),preds)
    np.save(osp.join(args.output_dir,model_fname+'_losses.npy'),losses)
    np.save(osp.join(args.output_dir,model_fname+'_val_losses.npy'),val_losses)
    plot_losses(losses, val_losses, model_fname, args.output_dir)
    make_plots(preds, ys, model_fname, args.output_dir)