from __future__ import division
from __future__ import unicode_literals
import numpy as np
import pandas as pd
import torch
import tqdm
import random
import statistics
import configparser
from distutils import util
# rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdEHTTools
# pytorch
import torch
from torch.nn import MSELoss
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
# sklearn
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# features
import os
import sys
import glob
from utils import *
from models import *
#fig

from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from IPython.display import SVG, display
import itertools
from matplotlib import gridspec
import seaborn as sns; sns.set(color_codes=True)


class EarlyStopping:
    
    def __init__(self, patience=7, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss

def isnan(x):
    return x!=x

def prepare_dataset(tar_list):
    cwd = os.getcwd()
    df_blank = pd.DataFrame({'smiles':[]})
    for dataset in tar_list:
        df0 = pd.read_csv(cwd+'/'+'data/'+dataset)
        df_blank =  pd.merge(df_blank, df0, on='smiles', how='outer')
    return df_blank

def prepare_data(train_df,task_name, num_atoms=False, use_hydrogen_bonding=False, use_acid_base = False):
   
    train_mol = []
    train_label = []
    
    for i in range(len(train_df)):
        train_mol.append(Chem.MolFromSmiles(train_df['smiles'][i]))
        train_label.append(train_df[task_name][i])

    train_all = [mol2vec(x, num_atoms=num_atoms, use_hydrogen_bonding=use_hydrogen_bonding,
                       use_acid_base = use_acid_base) for x in train_mol]
    
    for i, data in enumerate(train_all):
            data['y0'] = torch.tensor([train_label[i]], dtype=torch.float)
                
    return train_all

def train(loader, optimizer, model):
   
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    loss_all = 0
    output_vec = []
    true_vec = []
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fc([output[:,0]], [data['y0']])
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
        masks0 = isnan(data['y0']) 
        output_vec.append(output[:,0][~masks0])
        true_vec.append(data['y0'][~masks0])
    return loss_all / len(loader.dataset) , output_vec , true_vec

def loss_fc(output_vec, true_vec):
    
    criterion = torch.nn.MSELoss()
    mse_part = 0
    masks = dict()
    for x in range(0,len(true_vec)):
        masks[x] = isnan(true_vec[x]) 
        if true_vec[x][~masks[x]].nelement() == 0:
            loss1[x] = torch.sqrt(torch.tensor(1e-20))
            continue
        else: 
            mse_part += criterion(output_vec[x][~masks[x]],true_vec[x][~masks[x]])
    
    loss = torch.sqrt(mse_part)

    return loss

def test(loader, model):
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    loss_all = 0
    output_vec = []
    true_vec = []
        
    for data in loader:
        data = data.to(device)
        output = model(data)
        masks0 = isnan(data['y0']) 
        output_vec.append(output[:,0][~masks0])
        true_vec.append(data['y0'][~masks0])
        test_loss = loss_fc([output[:,0]], [data['y0']])
        loss_all += test_loss.item() * data.num_graphs
        
    return loss_all / len(loader.dataset) , output_vec , true_vec

def train_square(output_vec,true_vec):
    train_true_list = []
    train_predict_list = []
    
    for x in range(len(true_vec)):
        for n in range(len(true_vec[x])):
            train_true_list.append(true_vec[x][n].item())
    
    for x in range(len(output_vec)):
        for n in range(len(output_vec[x])):
            train_predict_list.append(output_vec[x][n].item())
    
    train_r2 = r2_score(train_true_list,train_predict_list)
   
    return train_r2 , train_predict_list , train_true_list

def test_rsquare(output_vec,true_vec):
    output_vec = output_vec[0].data.cpu().numpy()
    true_vec = true_vec[0].cpu().numpy()
    test_r2 = r2_score(true_vec,output_vec)
    
    return test_r2 , output_vec , true_vec

def val(loader, model):
   
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    loss_all = 0
    c = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        val_loss = loss_fc([output[:,0]], [data['y0']])
        loss_all += val_loss.item() * data.num_graphs
        c += data.y0.size(0)
    return loss_all/c

def clean_print(run, task_name, loss_vec):
    str1 = "Run %i : Total RMSE: %0.3f" %(run, loss_vec)
    str1 += " | %s Loss %0.3f" %(task_name, loss_vec)
    print(str1)
