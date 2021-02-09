import matplotlib.pyplot as plt
import numpy as np
import torch
import os.path as osp
from graph_data import GraphDataset

def make_hist(data, label, output_dir):
    plt.figure(figsize=(6,4.4))
    plt.hist(data, alpha = 0.5)
    plt.legend()
    plt.xlabel(label, fontsize=16)
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, label+'.pdf'))
    plt.close()

def get_x_input(data_dir):
    gdata = GraphDataset(root=data_dir, n_jets=100, n_events_merge=1)
    pt = []; eta = []; phi = []
    for i, d in enumerate(gdata):
        pt.append(gdata[0].x[:,0])
        eta.append(gdata[0].x[:,1])
        phi.append(gdata[0].x[:,2])
    torch.cat(pt); torch.cat(eta); torch.cat(phi)
    return (pt, "pt"), (eta, "eta"), (phi, "phi")

if __name__ == "__main__":
    import argparse;
    parser = argparse.ArgumentParser()
    parser.add_argument("--plt-input", action=store_true, help="plot pt eta phi", default=False, required=False)
    parser.add_argument("--data-dir", type=str, help="location of dataset", default="~/.energyflow/datasets", required=True)
    parser.add_argument("--save-dir", type=str, help="where to save figures", default="/energyflowvol/figures", required=True)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True) # make a folder for these graphs

    if args.plt_input:
        x_input = get_x_input(args.data_dir)
        for d in x_input:
            data = d[0]; label = d[1]
            make_hist(data, label, args.output_dir)