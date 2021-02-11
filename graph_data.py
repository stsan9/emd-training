import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
import itertools
import tables
import numpy as np
import pandas as pd
import energyflow as ef
import glob
from pyjet import cluster,DTYPE_PTEPM

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
        return return_list

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass


    def process(self):
        Js = []
        R = 0.4
        for raw_path in self.raw_paths:
            if self.lhco:
                # X, y = ef.qg_jets.load(self.n_jets, pad=False, cache_dir=self.root+'/raw')
                df = pd.read_hdf(raw_path, stop=self.n_events)
                all_events = df.values
                rows = all_events.shape[0]
                cols = all_events.shape[1]
                X = []
                # cluster jets and store info
                for i in range(rows):
                    pseudojets_input = np.zeros(len([x for x in all_events[i][::3] if x > 0]), dtype=DTYPE_PTEPM)
                    for j in range(cols // 3):
                        if (all_events[i][j*3]>0):
                            pseudojets_input[j]['pT'] = all_events[i][j*3]
                            pseudojets_input[j]['eta'] = all_events[i][j*3+1]
                            pseudojets_input[j]['phi'] = all_events[i][j*3+2]
                    sequence = cluster(pseudojets_input, R=1.0, p=-1)
                    jets = sequence.inclusive_jets()[:2] # leading 2 jets only
                    if len(jets) < 2: continue
                    for jet in jets: # for each jet get (px, py, pz, e)
                        if jet.pt < 200 or len(jets)<=1: continue
                        n_particles = len(jet)
                        particles = np.zeros((n_particles, 3))
                        # store all the particles of this jet
                        for p, part in enumerate(jet):
                            particles[p,:] = np.array([part.pt,
                                                       part.eta,
                                                       part.phi])
                        X.append(particles)
                X = np.array(X,dtype='O')
                # process jets
                Js = []
                for i,x in enumerate(X): 
                    if i >= self.n_jets: break
    #                 # ignore padded particles and removed particle id information
    #                 x = x[:,:3]
                    # center jet according to pt-centroid
                    yphi_avg = np.average(x[:,1:3], weights=x[:,0], axis=0)
                    x[:,1:3] -= yphi_avg
                    # mask out any particles farther than R=0.4 away from center (rare)
                    x = x[np.linalg.norm(x[:,1:3], axis=1) <= R]
                    if len(x) == 0: continue
                    # add to list
                    Js.append(x)
            else:
                # load quark and gluon jets
                X, y = ef.qg_jets.load(self.n_jets, pad=False, cache_dir=self.root+'/raw')
                # the jet radius for these jets
                # process jets
                Js = []
                for i,x in enumerate(X): 
                    if i >= self.n_jets: break
                    # ignore padded particles and removed particle id information
                    x = x[x[:,0] > 0,:3]
                    # center jet according to pt-centroid
                    yphi_avg = np.average(x[:,1:3], weights=x[:,0], axis=0)
                    x[:,1:3] -= yphi_avg
                    # mask out any particles farther than R=0.4 away from center (rare)
                    x = x[np.linalg.norm(x[:,1:3], axis=1) <= R]
                    # add to list
                    Js.append(x)
        self.n_jets = len(Js)
        jetpairs = [[i, j] for (i, j) in itertools.product(range(self.n_jets),range(self.n_jets))]
        datas = []
        for k, (i, j) in enumerate(jetpairs):
            try:
                emdval = ef.emd.emd(Js[i], Js[j], R=R)/ONE_HUNDRED_GEV
            except:
                import pdb; pdb.set_trace()
            Ei = np.sum(Js[i][:,0])
            Ej = np.sum(Js[j][:,0])
            jiNorm = Js[i].copy()
            jjNorm = Js[j].copy()
            jiNorm[:,0] = jiNorm[:,0]/Ei
            jjNorm[:,0] = jjNorm[:,0]/Ej
            jetpair = np.concatenate([jiNorm, jjNorm], axis=0)
            nparticles_i = len(Js[i])
            nparticles_j = len(Js[j])
            pairs = [[m, n] for (m, n) in itertools.product(range(0,nparticles_i),range(nparticles_i,nparticles_i+nparticles_j))]
            edge_index = torch.tensor(pairs, dtype=torch.long)
            edge_index=edge_index.t().contiguous()
            u = torch.tensor([[Ei/ONE_HUNDRED_GEV, Ej/ONE_HUNDRED_GEV]], dtype=torch.float)
            x = torch.tensor(jetpair, dtype=torch.float)
            y = torch.tensor([[emdval]], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y, u=u)
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