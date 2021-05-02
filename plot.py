"""
File: plot.py
Plot either input distributions or graphs of emd-network output

Example (visualize model output):
python plot.py --plot-nn-eval \
    --model "SymmetricDDEdgeNet" \
    --data-dir "/energyflowvol/eval_lhco_data_150" \
    --save-dir "/energyflowvol/figures/symmDD_lhco_model_new_loss_1k" \
    --model-dir "/energyflowvol/symmDD_lhco_model_1k_new_loss" \
    --n-jets 150 \
    --n-events-merge 500 \
    --remove-dupes
"""
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import torch
import math
import tqdm
from pathlib import Path
from torch.utils.data import random_split
from graph_data import GraphDataset, ONE_HUNDRED_GEV
from torch_geometric.data import Data, DataLoader

# personal code
import models
from process_util import remove_dupes

def make_hist(data, label, save_dir):
    plt.figure(figsize=(6,4.4))
    plt.hist(data)
    plt.legend()
    plt.xlabel(label, fontsize=16)
    plt.tight_layout()
    plt.savefig(osp.join(save_dir, label+'.pdf'))
    plt.close()

def get_y_output(gdata):
    y = []
    for d in gdata:
        y.append(d[0].y[0])
    y = torch.cat(y)
    return y

def get_x_input(gdata):
    pt = []; eta = []; phi = []
    for d in gdata:
        pt.append(d[0].x[:,0])
        eta.append(d[0].x[:,1])
        phi.append(d[0].x[:,2])
    pt = torch.cat(pt)
    eta = torch.cat(eta)
    phi = torch.cat(phi)
    return (pt, "pt"), (eta, "eta"), (phi, "phi")

def make_plots(preds, ys, model_fname, save_dir):

    # largest y-value rounded to nearest 100
    max_range = round(np.max(ys),-2)
    
    diffs = (preds-ys)
    rel_diffs = diffs[ys>0]/ys[ys>0]

    # plot figures
    plt.rcParams['figure.figsize'] = (4,4)
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['font.family'] = 'serif'

    fig, ax = plt.subplots(figsize =(5, 5)) 
    plt.hist(ys, bins=np.linspace(0, max_range , 101),label='True', alpha=0.5)
    plt.hist(preds, bins=np.linspace(0, max_range, 101),label = 'Pred.', alpha=0.5)
    plt.legend()
    ax.set_xlabel('EMD [GeV]') 
    fig.savefig(osp.join(save_dir,model_fname+'_EMD.pdf'))
    fig.savefig(osp.join(save_dir,model_fname+'_EMD.png'))

    fig, ax = plt.subplots(figsize =(5, 5)) 
    plt.hist(diffs, bins=np.linspace(-200, 200, 101))
    ax.set_xlabel(f'EMD diff. [GeV], std: {"{:.3e}".format(np.std(diffs))}, mean: {"{:.3e}".format(np.mean(diffs))}')  
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_diff.pdf'))
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_diff.png'))

    fig, ax = plt.subplots(figsize =(5, 5)) 
    plt.hist(rel_diffs, bins=np.linspace(-1, 1, 101))
    ax.set_xlabel(f'EMD rel. diff., std: {"{:.3e}".format(np.std(rel_diffs))}, mean: {"{:.3e}".format(np.mean(rel_diffs))}')  
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_rel_diff.pdf'))
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_rel_diff.png'))

    fig, ax = plt.subplots(figsize =(5, 5)) 
    x_bins = np.linspace(0, max_range, 101)
    y_bins = np.linspace(0, max_range, 101)
    plt.hist2d(ys, preds, bins=[x_bins,y_bins])
    ax.set_xlabel('True EMD [GeV]')  
    ax.set_ylabel('Pred. EMD [GeV]')
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_corr.pdf'))
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_corr.png'))


if __name__ == "__main__":
    import argparse;
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-input", action='store_true', help="plot pt eta phi", default=False, required=False)
    parser.add_argument("--plot-nn-eval", action='store_true', help="plot graphs for evaluating emd nn's", default=False, required=False)
    parser.add_argument("--model", choices=['EdgeNet', 'DynamicEdgeNet','DeeperDynamicEdgeNet','DeeperDynamicEdgeNetPredictFlow',
                                            'DeeperDynamicEdgeNetPredictEMDFromFlow','SymmetricDDEdgeNet'], 
                        help="Model name", required=True)
    parser.add_argument("--data-dir", type=str, help="location of dataset", default="~/.energyflow/datasets", required=True)
    parser.add_argument("--save-dir", type=str, help="where to save figures", default="/energyflowvol/figures", required=True)
    parser.add_argument("--model-dir", type=str, help="path to folder with model", default="/energyflowvol/models2/", required=False)
    parser.add_argument("--n-jets", type=int, help="number of jets", required=False, default=150)
    parser.add_argument("--n-events-merge", type=int, help="number of events to merge", required=False, default=500)
    parser.add_argument("--remove-dupes", action="store_true", help="remove dupes in data with different jet ordering", required=False)
    args = parser.parse_args()

    Path(args.save_dir).mkdir(exist_ok=True) # make a folder for these graphs
    gdata = GraphDataset(root=args.data_dir, n_jets=args.n_jets, n_events_merge=args.n_events_merge)

    if args.plot_input:
        x_input = get_x_input(gdata)
        for d in x_input:
            data = d[0]; label = d[1]
            make_hist(data.numpy(), label, args.save_dir)

    if args.plot_nn_eval:
        if args.model_dir is None:
            exit("No args.model-dir not specified")

        # load all data into memory at once
        test_dataset = []
        for g in gdata:
            test_dataset += g
        if args.remove_dupes:
            test_dataset = remove_dupes(test_dataset)

        # load in model
        input_dim = 4
        big_dim = 32
        bigger_dim = 128
        global_dim = 2
        output_dim = 1
        batch_size=100
        device = 'cuda:0'
        model_class = getattr(models, args.model)
        model = model_class(input_dim=input_dim, big_dim=big_dim, bigger_dim=bigger_dim, 
                            global_dim=global_dim, output_dim=output_dim).to(device)
        model_fname = args.model
        modpath = osp.join(args.model_dir,model_fname+'.best.pth')
        try:
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(modpath, map_location=torch.device('cuda')))
            else:
                model.load_state_dict(torch.load(modpath, map_location=torch.device('cpu')))
        except:
            exit("No model")
        
        # get test dataset
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
        test_samples = len(test_dataset)
        
        # save folder
        eval_folder = 'eval'
        if not args.remove_dupes:
            eval_folder += '_dupes'
        eval_dir = osp.join(args.save_dir, eval_folder)
        Path(eval_dir).mkdir(exist_ok=True)

        # evaluate model
        ys = []
        preds = []
        diffs = []
        t = tqdm.tqdm(enumerate(test_loader),total=test_samples/batch_size)
        model.eval()
        for i, data in t:
            data.to(device)
            out = model(data)
            if model_fname == "SymmetricDDEdgeNet":
                out = out[0]    # toss unecessary terms
            ys.append(data.y.cpu().numpy().squeeze()*ONE_HUNDRED_GEV)
            preds.append(out.cpu().detach().numpy().squeeze()*ONE_HUNDRED_GEV)
        ys = np.concatenate(ys)   
        preds = np.concatenate(preds)   

        # plot results
        make_plots(preds, ys, model_fname, eval_dir)