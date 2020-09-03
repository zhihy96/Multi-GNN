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

config = configparser.ConfigParser()
config.read('config.cfg')
seed = int(config['parameters']['seed'])
cwd = os.getcwd()
seed = int(config['parameters']['seed'])
nbr_task = int(config['parameters']['number_tasks'])
dlist = config['parameters']['task_names'].split(',',nbr_task)
dfnew = prepare_dataset(dlist)

# split train/test 8:2 with random seed
train_df = dfnew.sample(frac=0.8, random_state=seed)
test_df  = dfnew[~dfnew.index.isin(train_df.index)]
train_df = train_df.reset_index()
test_df  = test_df.reset_index()
task_name = dlist[0].split('.')[0]

num_atoms            = util.strtobool(config['parameters']['num_atoms'])
use_hydrogen_bonding = util.strtobool(config['parameters']['use_hydrogen_bonding'])
use_acid_base       = util.strtobool(config['parameters']['use_acid_base'])
n_splits             = int(config['parameters']['n_splits'])
dim                  = int(config['parameters']['dim'])
n_epochs             = int(config['parameters']['n_epochs'])
n_batchs             = int(config['parameters']['batch'])
n_iterations         = int(config['parameters']['n_iterations'])
patience             = int(config['parameters']['patience'])
patience_early       = int(config['parameters']['patience_early'])
lr_decay             = float(config['parameters']['lr_decay'])

test_all = prepare_data(test_df, task_name,
                      num_atoms = num_atoms,
                      use_hydrogen_bonding=use_hydrogen_bonding,
                      use_acid_base=use_acid_base)
test_loader = DataLoader(test_all, batch_size=len(test_df), shuffle=False, drop_last=False)
for data in test_loader:
    n_features = data.x.size(1)
    break

print('-'*60)
print("Number of features to be used: %i" %n_features)
print('-'*60)

modelstr = config['parameters']['model']

if modelstr == 'GIAN':
    model0 = gian.GIAN(n_features, n_outputs=1, dim=dim)

elif modelstr == 'GIAT':
    model0 = giat.GIAT(n_features, n_outputs=1, dim=dim)

elif modelstr == 'SGCA':
    model0 = sgca.SGCA(n_features, n_outputs=1, dim=dim)
    
    
else:
    print('You did not put a correct model')
    print('Available models: GIAN, GIAT, SGCA')
    print('')
    print('Try Again')
    sys.exit()
    
    
cv = KFold(n_splits=n_splits)
run = 1
num_atoms = False
use_hydrogen_bonding = False
use_acid_base = False
run = 1
train_rmse = 0
train_r2_total = 0
test_rmse = 0
test_r2_total = 0

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

train_all = prepare_data(train_df, task_name, num_atoms=num_atoms, use_hydrogen_bonding=use_hydrogen_bonding, 
                 use_acid_base=use_acid_base)

path = os.path.join('.','model_checkpoint',modelstr+'.pk')
model0.load_state_dict(torch.load(path))

for train_index, val_index in cv.split(train_all):
    
    
    model = model0
    model = model.to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay,
                                                           verbose=False)
    early_stopping = EarlyStopping(patience=patience_early, verbose=False)

    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(val_index)
    train_loader = DataLoader(train_all, batch_size=n_batchs,  sampler=train_sampler)
    val_loader = DataLoader(train_all, batch_size=n_batchs,  sampler=valid_sampler)


    for epoch in range(1, n_epochs):
        train_loss , train_predict , train_true = train(train_loader, optimizer, model)
        val_loss   = val(val_loader, model)
        print('epoch %i: normalized train loss %0.2f val loss %0.2f' %(epoch, train_loss, val_loss), end="\r")
        scheduler.step(train_loss)
        early_stopping(val_loss)

        if early_stopping.early_stop:
            print("Early stopping")
            break
            
    # saving the model at the end of each CV
    torch.save(model0.state_dict(), 'model_checkpoint/%s%i.pk' %(task_name,run))
    
    #train rmse , rsquare , predict , label
    train_r2 , train_predict_np , train_true_np = train_square(train_predict,train_true)
    
    train_rmse += train_loss
    train_r2_total += train_r2
    
    # test rmse , rsquare , predict , label
    test_loss , test_predict , test_true = test(test_loader, model)
    test_r2 , test_predict_np , test_true_np= test_rsquare(test_predict,test_true)
    
    test_rmse += test_loss
    test_r2_total += test_r2
    
    #processce
    clean_print(run, task_name, test_loss)
    run += 1
    train_rmse_mean = train_rmse / n_splits
    train_r2_mean = train_r2_total / n_splits
    test_rmse_mean = test_rmse / n_splits
    test_r2_mean = test_r2_total / n_splits
    
print(" ")
print('-'*20+'Model Cross-Validation'+'-'*20)
print('train rmse:', train_rmse_mean)
print('train r2:',train_r2_mean)
print('test rmse:', test_rmse_mean)
print('test r2:',test_r2_mean)


from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from IPython.display import SVG, display
import itertools
from matplotlib import gridspec
import seaborn as sns; sns.set(color_codes=True)


plt.figure(figsize=(6, 6),dpi=600)

#gs = gridspec.GridSpec(1, 4)

x, y = pd.Series(train_true_np, name="Experimental Values"), pd.Series(train_predict_np, name='Predicted Values')
x2, y2 = pd.Series(test_true_np,name='Experimental Values'), pd.Series(test_predict_np,name='Predicted Values')
sns.set_style('whitegrid')
ax1 = plt.subplot(111)
ax1.axis([3.9,9.5,3.9,9.5])
ax1.set_title(label=modelstr, loc='center', pad=5 ,fontdict={'fontsize':12})
sns.regplot(x=x, y=y, color="blue", 
             label='train r-square = '+str(np.around(train_r2_mean,decimals=3)),
             ax = ax1 ,ci = None, scatter_kws = {'s':4},line_kws={'linewidth':1,'linestyle':'--'})
sns.regplot(x = x2, y = y2, color = "red", 
            label='test  r-square = '+str(np.around(test_r2_mean,decimals=3)),
            ax = ax1 , ci = None, scatter_kws = {'s':4},line_kws={'linewidth':1})

ax1.legend(loc='lower right',frameon=True,shadow=True,edgecolor='black')

plt.text(4,8.2,'RMSE = '+str(np.around(test_rmse_mean,decimals=3)),fontsize=15)
plt.savefig(modelstr+"1.png",bbox_inches = 'tight',dpi=600)
plt.show()
plt.close()


model0.load_state_dict(torch.load(path))

dfpred = prepare_dataset(['predict.csv'])

def pred(model):
    pred_X = prepare_data(dfpred,task_name='pic50' ,
                              num_atoms=num_atoms,
                              use_hydrogen_bonding=use_hydrogen_bonding,
                              use_acid_base=use_acid_base)

    pred_loader = DataLoader(pred_X, batch_size=len(dfpred), shuffle=False, drop_last=False)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    for data in pred_loader:
        data = data.to(device)
        output = model0(data)
        output = output.data.cpu().numpy()
        output = list(output.ravel())
        print(output)
        
    return output
        
predict = pred(model0)
