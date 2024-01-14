import numpy as np
import os
from utils import generate_mask
from run_tdm import run_TDM

missing_prop = 0.3
missing_type = 'MCAR' # Choosing from MAR, MNARL, MNARQ, MCAR
data = np.load("datasets/seeds_complete.npy")
X_true = data
mask = generate_mask(X_true, missing_prop, missing_type)
X_missing = np.copy(X_true)
X_missing[mask.astype(bool)] = np.nan

niter = 10
batchsize = 64
lr = 1e-2
report_interval = 100
network_depth = 3
network_width = 2
args = {'out_dir': f'./demo_duong', 'niter': niter,
 'batchsize': batchsize, 'lr': lr, 'network_width': network_width, 'network_depth': network_depth, 'report_interval': report_interval}


run_TDM(X_missing, args, X_true)