import numpy as np
import matplotlib.pyplot as plt
import torch
import os.path as osp
import os
import sys
import tqdm

from graph_data import GraphDataset, ONE_HUNDRED_GEV
from torch_geometric.data import DataLoader
from process_util import pair_dupes

def col(matrix, i):
    return [row[i] for row in matrix]

def get_emds(model, loader, length, batch_size):
    ys = []
    preds = []

    t = tqdm.tqdm(enumerate(loader),total=length/batch_size)
    model.eval()
    for i, data in t:
        data.to(device)
        true_emd = data.y
        learn_emd = model(data)

        ys.append(true_emd.cpu().numpy().squeeze()*ONE_HUNDRED_GEV)
        preds.append(learn_emd.cpu().detach().numpy().squeeze()*ONE_HUNDRED_GEV)
    ys = np.array(ys)
    preds = np.array(preds)
    return ys, preds

def make_symm_diff_plots(pred_diffs, avg_diffs, model_fname, output_dir):
    max_range = max(max(pred_diffs), max(avg_diffs))
    fig, ax = plt.subplots(figsize =(5, 5)) 
    plt.hist(pred_diffs, bins=np.linspace(0, max_range, 101),label = 'Emd_1 - Emd_2', alpha=0.5)
    plt.hist(avg_diffs, bins=np.linspace(0, max_range, 101),label='Avgs. - Trues', alpha=0.5)
    plt.legend()
    ax.set_xlabel('EMD Differences of Symmetrical Pairs') 
    fig.savefig(osp.join(output_dir,model_fname+'_symm_diff.pdf'))
    fig.savefig(osp.join(output_dir,model_fname+'_symm_diff.png'))

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
    parser.add_argument("--lhco", action='store_true', help="Using lhco dataset (diff processing)", default=False, required=False)
    parser.add_argument("--n-jets", type=int, help="number of jets", required=False, default=100)
    parser.add_argument("--n-events-merge", type=int, help="number of events to merge", required=False, default=1)
    parser.add_argument("--batch-size", type=int, help="batch size", required=False, default=100)
    args = parser.parse_args()

    # define model
    import models
    model_class = getattr(models, args.model)
    input_dim = 4
    big_dim = 32
    bigger_dim = 128
    global_dim = 2
    output_dim = 1
    batch_size = args.batch_size
    device = 'cuda:0'
    model_fname = args.model
    modpath = osp.join(args.output_dir,model_fname+'.best.pth')
    model = model_class(input_dim=input_dim, big_dim=big_dim, bigger_dim=bigger_dim, 
                        global_dim=global_dim, output_dim=output_dim).to(device)
    try:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(modpath, map_location=torch.device('cuda')))
        else:
            model.load_state_dict(torch.load(modpath, map_location=torch.device('cpu')))
    except:
        exit("Evaluating on non-existent model")

    if args.model == 'SymmetricDDEdgeNet':
        model = model.EdgeNet

    # process dataset and create loaders
    gdata = GraphDataset(root=args.input_dir, n_jets=args.n_jets, n_events_merge=args.n_events_merge, lhco=args.lhco)
    bag = []
    for g in gdata:
        bag += g
    bag = pair_dupes(bag)
    J1s = col(bag, 0)
    J2s = col(bag, 1)
    J1_loader = DataLoader(J1s, batch_size=batch_size, pin_memory=True, shuffle=False)
    J2_loader = DataLoader(J2s, batch_size=batch_size, pin_memory=True, shuffle=False)

    # get emds
    true_emd_1, pred_emd_1 = get_emds(model, J1_loader, len(J1s), args.batch_size)
    true_emd_2, pred_emd_2 = get_emds(model, J2_loader, len(J2s), args.batch_size)

    # sanity check
    if not np.array_equal(true_emd_1, true_emd_2):
        exit("Messed up ordering of J1 and J2")

    # calculate differences for plotting
    pred_diffs = abs(pred_emd_1 - pred_emd_2)
    pred_avg = (pred_emd_1 + pred_emd_2) / 2
    avg_diffs = abs(pred_avg - true_emd_1)

    # make and save plots
    eval_dir = osp.join(args.output_dir, 'eval')
    os.makedirs(eval_dir,exist_ok=True)
    make_symm_diff_plots(pred_diffs, avg_diffs, model_fname, eval_dir)