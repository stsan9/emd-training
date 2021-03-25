import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
import itertools
import tables
import numpy as np
import energyflow as ef
import glob
from process_util import jet_particles
from natsort import natsorted
from sys import exit

ONE_HUNDRED_GEV = 100.0

class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, 
                 n_jets=1000, n_events_merge=100, n_events=1000, lhco=False):
        self.n_jets = n_jets
        self.n_events_merge = n_events_merge
        self.n_events = n_events
        self.lhco = lhco
        super(GraphDataset, self).__init__(root, transform, pre_transform) 


    @property
    def raw_file_names(self):
        return ['events_LHCO2020_backgroundMC_Pythia.h5']

    @property
    def processed_file_names(self):
        """
        Returns a list of all the files in the processed files directory
        """
        proc_list = glob.glob(osp.join(self.processed_dir, 'data_*.pt'))
        n_files = int(self.n_jets*self.n_jets/self.n_events_merge)
        return_list = list(map(osp.basename, proc_list))[:n_files]
        return natsorted(return_list)

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        Js = []
        R = 0.4
        for raw_path in self.raw_paths:
            # load jet-particles dataset
            if self.lhco:
                print("Loading LHCO Dataset")
                X = jet_particles(raw_path, self.n_events)
            else:
                print("Loading QG Dataset")
                X, _ = ef.qg_jets.load(self.n_jets, pad=False, cache_dir=self.root+'/raw')
            
            # clean and store list of jets as particles (pt, eta, phi)
            Js = []
            jet_ctr = 0
            for x in X: 
                if not self.lhco:
                    # ignore padded particles and removed particle id information
                    x = x[x[:,0] > 0,:3]
                # center jet according to pt-centroid
                yphi_avg = np.average(x[:,1:3], weights=x[:,0], axis=0)
                x[:,1:3] -= yphi_avg
                # mask out any particles farther than R=0.4 away from center (rare)
                x = x[np.linalg.norm(x[:,1:3], axis=1) <= R]
                # add to list
                if len(x) == 0: continue
                Js.append(x)
                # stop when n_jets stored
                jet_ctr += 1
                if jet_ctr == self.n_jets: break

        # calc emd between all jet pairs and save datum
        jetpairs = [[i, j] for (i, j) in itertools.product(range(self.n_jets),range(self.n_jets))]
        datas = []
        for k, (i, j) in enumerate(jetpairs):    
            if k % (len(jetpairs) // 20) == 0:
                print(f'Generated: {k}/{len(jetpairs)}')
            emdval, G = ef.emd.emd(Js[i], Js[j], R=R, return_flow=True)
            emdval = emdval/ONE_HUNDRED_GEV
            G = G/ONE_HUNDRED_GEV
            Ei = np.sum(Js[i][:,0])
            Ej = np.sum(Js[j][:,0])
            jiNorm = np.zeros((Js[i].shape[0],Js[i].shape[1]+1)) # add a field
            jjNorm = np.zeros((Js[j].shape[0],Js[j].shape[1]+1)) # add a field
            jiNorm[:,:3] = Js[i].copy()
            jjNorm[:,:3] = Js[j].copy()
            jiNorm[:,0] = jiNorm[:,0]/Ei
            jjNorm[:,0] = jjNorm[:,0]/Ej
            jiNorm[:,3] = -1*np.ones((Js[i].shape[0]))
            jjNorm[:,3] = np.ones((Js[j].shape[0]))
            jetpair = np.concatenate([jiNorm, jjNorm], axis=0)
            nparticles_i = len(Js[i])
            nparticles_j = len(Js[j])
            pairs = [[m, n] for (m, n) in itertools.product(range(0,nparticles_i),range(nparticles_i,nparticles_i+nparticles_j))]
            edge_index = torch.tensor(pairs, dtype=torch.long)
            edge_index = edge_index.t().contiguous()
            u = torch.tensor([[Ei/ONE_HUNDRED_GEV, Ej/ONE_HUNDRED_GEV]], dtype=torch.float)
            edge_y = torch.tensor([[G[m,n-nparticles_i] for m, n in pairs]], dtype=torch.float)
            edge_y = edge_y.t().contiguous()

            x = torch.tensor(jetpair, dtype=torch.float)
            y = torch.tensor([[emdval]], dtype=torch.float)
            
            data = Data(x=x, edge_index=edge_index, y=y, u=u, edge_y=edge_y)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            datas.append([data])                  
            if k%self.n_events_merge == self.n_events_merge-1:
                datas = sum(datas,[])
                torch.save(datas, osp.join(self.processed_dir, 'data_{}.pt'.format(k)))
                datas=[]
            
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        return data
