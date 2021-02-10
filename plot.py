import matplotlib.pyplot as plt
import numpy as np
import torch
import os.path as osp
from graph_data import GraphDataset, ONE_HUNDRED_GEV
from pathlib import Path
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from torch.utils.data import random_split
import models
import tqdm

def make_hist(data, label, save_dir):
    plt.figure(figsize=(6,4.4))
    plt.hist(data)
    plt.legend()
    plt.xlabel(label, fontsize=16)
    plt.tight_layout()
    plt.savefig(osp.join(save_dir, label+'.pdf'))
    plt.close()

def get_x_input(gdata):
    pt = []; eta = []; phi = []
    for d in gdata:
        pt.append(d[0].x[:,0])
        eta.append(d[0].x[:,1])
        phi.append(d[0].x[:,2])
    torch.cat(pt); torch.cat(eta); torch.cat(phi)
    return (pt, "pt"), (eta, "eta"), (phi, "phi")

@torch.no_grad()
def test(model,loader,total,batch_size):
    model.eval()
    
    mse = nn.MSELoss(reduction='mean')

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        data = data.to(device)
        batch_output = model(data)
        batch_loss_item = mse(batch_output, data.y).item()
        sum_loss += batch_loss_item
        t.set_description("loss = %.5f" % (batch_loss_item))
        t.refresh() # to show immediately the update

    return sum_loss/(i+1)

def eval_nn(model, test_dataset, model_fname, save_dir):
    batch_size=100
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_loader.collate_fn = collate
    test_samples = len(test_dataset)

    # evaluate model
    ys = []
    preds = []
    diffs = []
    t = tqdm.tqdm(enumerate(test_loader),total=test_samples/batch_size)
    for i, data in t:
        data.to(device)
        ys.append(data.y.cpu().numpy().squeeze()*ONE_HUNDRED_GEV)
        preds.append(model(data).cpu().detach().numpy().squeeze()*ONE_HUNDRED_GEV)
    ys = np.concatenate(ys)   
    preds = np.concatenate(preds)   
    diffs = (preds-ys)
    rel_diffs = diffs[ys>0]/ys[ys>0]
    
    # plot figures
    plt.rcParams['figure.figsize'] = (4,4)
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['font.family'] = 'serif'

    fig, ax = plt.subplots(figsize =(5, 5)) 
    plt.hist(ys, bins=np.linspace(0, 300, 101),label='True', alpha=0.5)
    plt.hist(preds, bins=np.linspace(0, 300, 101),label = 'Pred.', alpha=0.5)
    plt.legend()
    ax.set_xlabel('EMD [GeV]') 
    fig.savefig(osp.join(save_dir,model_fname+'_EMD.pdf'))
    fig.savefig(osp.join(save_dir,model_fname+'_EMD.png'))

    fig, ax = plt.subplots(figsize =(5, 5)) 
    plt.hist(diffs, bins=np.linspace(-200, 200, 101))
    ax.set_xlabel('EMD diff. [GeV]')  
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_diff.pdf'))
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_diff.png'))

    fig, ax = plt.subplots(figsize =(5, 5)) 
    plt.hist(rel_diffs, bins=np.linspace(-1, 1, 101))
    ax.set_xlabel('EMD rel. diff.')  
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_rel_diff.pdf'))
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_rel_diff.png'))

    fig, ax = plt.subplots(figsize =(5, 5)) 
    x_bins = np.linspace(0, 300, 101)
    y_bins = np.linspace(0, 300, 101)
    plt.hist2d(ys, preds, bins=[x_bins,y_bins])
    ax.set_xlabel('True EMD [GeV]')  
    ax.set_ylabel('Pred. EMD [GeV]')
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_corr.pdf'))
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_corr.png'))


if __name__ == "__main__":
    import argparse;
    parser = argparse.ArgumentParser()
    parser.add_argument("--plt-input", action='store_true', help="plot pt eta phi", default=False, required=False)
    parser.add_argument("--plt-nn-eval", action='store_true', help="plot graphs for evaluating emd nn's", default=False, required=False)
    parser.add_argument("--model", choices=['EdgeNet', 'DynamicEdgeNet', 'DeeperDynamicEdgeNet'], 
                        help="Model name", required=False, default='DeeperDynamicEdgeNet')
    parser.add_argument("--model-dir", type=str, help="path to folder with model", default="/energyflowvol/models2/", required=False)
    parser.add_argument("--data-dir", type=str, help="location of dataset", default="~/.energyflow/datasets", required=True)
    parser.add_argument("--save-dir", type=str, help="where to save figures", default="/energyflowvol/figures", required=True)
    parser.add_argument("--n-jets", type=int, help="number of jets", required=False, default=100)
    parser.add_argument("--n-events-merge", type=int, help="number of events to merge", required=False, default=1)
    args = parser.parse_args()

    Path(args.save_dir).mkdir(exist_ok=True) # make a folder for these graphs
    gdata = GraphDataset(root=args.data_dir, n_jets=args.n_jets, n_events_merge=args.n_events_merge)

    if args.plt_input:
        x_input = get_x_input(gdata)
        for d in x_input:
            data = d[0]; label = d[1]
            make_hist(data, label, args.save_dir)

    if args.plt_nn_eval:
        # load in model
        input_dim = 3
        big_dim = 32
        bigger_dim = 128
        global_dim = 2
        output_dim = 1
        fulllen = len(gdata)
        tv_frac = 0.10
        tv_num = math.ceil(fulllen*tv_frac)
        batch_size = args.batch_size
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
        def collate(items):
            l = sum(items, [])
            return Batch.from_data_list(l)
        _, _, test_dataset = random_split(gdata, [fulllen-2*tv_num,tv_num,tv_num])

        eval_nn(model, test_dataset, model_fname, args.save_dir)