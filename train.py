#!/usr/bin/env python
import os
import sys
import math
import tqdm
import torch
import random
import logging
import numpy as np
import os.path as osp
import torch.nn as nn
import energyflow as ef
import matplotlib.pyplot as plt
from torch_scatter import scatter_add
from torch.utils.data import random_split
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch

import models
from plot import make_plots
from loss_util import LossFunction, get_emd
from process_util import remove_dupes, pair_dupes
from graph_data import GraphDataset, ONE_HUNDRED_GEV

plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'serif'
torch.manual_seed(0)
np.random.seed(0)

@torch.no_grad()
def test(model, loader, total, batch_size, loss_ftn_obj):
    model.eval()
    
    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        data = data.to(device)
        batch_output = model(data)

        if loss_ftn_obj.name == 'predict_flow':
            batch_loss = loss_ftn_obj(data, batch_output)
        elif loss_ftn_obj.name == 'symm_loss_1':
            # loss = mse(y, pred) + mse(emd1, emd2)
            batch_output, emd_1, emd_2 = batch_output
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, data.y, emd_1, emd_2)
        elif loss_ftn_obj.name == 'symm_loss_2':
            # loss = mse(y, pred) + lam * pred^2
            batch_output, _, _ = batch_output
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, data.y)
        else:
            batch_output = model(data)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, data.y)    # mse

        batch_loss_item = batch_loss.item()
        sum_loss += batch_loss_item
        t.set_description("loss = %.5f" % (batch_loss_item))
        t.refresh() # to show immediately the update

    return sum_loss/(i+1)

def train(model, optimizer, loader, total, batch_size, loss_ftn_obj):
    model.train()
    
    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        data = data.to(device)
        optimizer.zero_grad()
        batch_output = model(data)

        if loss_ftn_obj.name == 'predict_flow':
            batch_loss = loss_ftn_obj(data, batch_output)
        elif loss_ftn_obj.name == 'symm_loss_1':
            # loss = mse(y, pred) + mse(emd1, emd2)
            batch_output, emd_1, emd_2 = batch_output
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, data.y, emd_1, emd_2)
        elif loss_ftn_obj.name == 'symm_loss_2':
            # loss = mse(y, pred) + lam * pred^2
            batch_output, _, _ = batch_output
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, data.y)
        else:
            batch_output = model(data)
            batch_loss = loss_ftn_obj.loss_ftn(batch_output, data.y)    # mse

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

    # directories
    parser.add_argument("--output-dir", type=str, help="Output directory for models and plots.", required=False, 
                        default='models2/')
    parser.add_argument("--input-dir", type=str, help="Input directory for datasets.", required=False,
                        default='/energyflowvol/datasets/')
    # model
    parser.add_argument("--model", choices=['EdgeNet', 'DynamicEdgeNet','DeeperDynamicEdgeNet','DeeperDynamicEdgeNetPredictFlow',
                                            'DeeperDynamicEdgeNetPredictEMDFromFlow','SymmetricDDEdgeNet'], 
                        help="Model name", required=False, default='DeeperDynamicEdgeNet')
    # loss
    parser.add_argument("--loss", choices=['symm_loss_1', 'symm_loss_2', 'mse', 'predict_flow'], help="loss function choice", required=True)
    parser.add_argument("--lam1", type=float, help="lambda1 for predict_flow (emd term) or symm_loss_2", default=1, required=False)
    parser.add_argument("--lam2", type=float, help="lambda2 for predict flow (fij loss term)", default=100, required=False)
    # dataset
    parser.add_argument("--lhco", action='store_true', help="Using lhco dataset (diff processing)", default=False, required=False)
    parser.add_argument("--n-jets", type=int, help="number of jets", required=False, default=100)
    parser.add_argument("--n-events-merge", type=int, help="number of events to merge", required=False, default=1)
    parser.add_argument("--remove-dupes", action="store_true", help="remove data that had the same jet pair in different order (leave one ver)", required=False, default=False)
    parser.add_argument("--pair-dupes", action="store_true", help="pair data that use the same jet pair", required=False, default=False)
    # hyperparams
    parser.add_argument("--batch-size", type=int, help="batch size", required=False, default=100)
    parser.add_argument("--n-epochs", type=int, help="number of epochs", required=False, default=100)
    parser.add_argument("--patience", type=int, help="patience for early stopping", required=False, default=10)
    args = parser.parse_args()

    # create output directory
    os.makedirs(args.output_dir,exist_ok=True)

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
    lr = 0.001
    device = 'cuda:0'
    model_fname = args.model
    modpath = osp.join(args.output_dir,model_fname+'.best.pth')
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
    loss_ftn_obj = LossFunction(args.loss, args.lam1, args.lam2)

    # load data
    logging.debug("Loading dataset...")
    gdata = GraphDataset(root=args.input_dir, n_jets=args.n_jets, n_events_merge=args.n_events_merge, lhco=args.lhco)
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
    best_valid_loss = test(model, valid_loader, valid_samples, batch_size, loss_obj)
    losses = []
    val_losses = []
    for epoch in range(0, n_epochs):
        loss = train(model, optimizer, train_loader, train_samples, batch_size, loss_obj)
        losses.append(loss)
        valid_loss = test(model, valid_loader, valid_samples, batch_size, loss_obj)
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

    # test model
    model.load_state_dict(torch.load(modpath))
    ys = []
    preds = []
    diffs = []
    
    t = tqdm.tqdm(enumerate(test_loader),total=test_samples/batch_size)
    model.eval()
    for i, data in t:
        data.to(device)
        if loss_ftn_obj.name == 'predict_flow':
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