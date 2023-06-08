#!/usr/bin/env python
# coding: utf-8

print('mazaak chal raha hai kya?')


import ase
import yaml
import time
import copy
import joblib
import pickle
import numpy as np
import datetime
from ase import Atoms, io, build

import matplotlib.pyplot as plt

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# structrepgen
from structrepgen.utils.dotdict import dotdict
from structrepgen.utils.utils import torch_device_select

from multi_vae import CVAE, Trainer
#from cvae_uncon import CVAE, Trainer

# ### CONFIG

# In[38]:

CONFIG = {}

CONFIG['gpu'] = True
CONFIG['params'] = {}
CONFIG['params']['seed'] = 42
CONFIG['params']['split_ratio'] = 0.2

CONFIG['params']['input_dim'] = 708 # input dimmension
CONFIG['params']['input_dim1'] = 600 # input dimmension
CONFIG['params']['input_dim2'] = 7 # input dimmension
CONFIG['params']['input_dim3'] = 101 # input dimmension

CONFIG['params']['hidden_dim'] = 512
CONFIG['params']['hidden_dim1'] = 512
CONFIG['params']['hidden_dim2'] = 8
CONFIG['params']['hidden_dim3'] = 64

CONFIG['params']['output_dim'] = 708

CONFIG['params']['latent_dim'] = 64
CONFIG['params']['hidden_layers'] = 5
CONFIG['params']['y_dim'] = 1
CONFIG['params']['batch_size'] = 512
CONFIG['params']['n_epochs'] = 1500
CONFIG['params']['lr'] = 2e-4
CONFIG['params']['final_decay'] = 0.2
CONFIG['params']['weight_decay'] = 0.001
CONFIG['params']['verbosity'] = 10
CONFIG['params']['kl_weight'] = 1e-6
CONFIG['params']['mc_kl_loss'] = False
CONFIG['params']['act_fn'] = "ELU"

'''

CONFIG['data_x_path'] = 'MP_data_csv/mp_20_raw_train_dist_mat.pt'
CONFIG['data_ead_path'] = 'MP_data_csv/mp_20_raw_train_ead_mat.pt'
CONFIG['composition_path'] = 'MP_data_csv/mp_20_raw_train_composition_mat.pt'
CONFIG['cell_path'] = 'MP_data_csv/mp_20_raw_train_cell_mat.pt'
CONFIG['data_y_path'] = "MP_data_csv/mp_20/raw_train/targets.csv"

CONFIG['unprocessed_path'] = 'MP_data_csv/mp_20_raw_train_unprocessed.txt'

# CONFIG['model_path'] = 'saved_models/cvae_saved.pt'
CONFIG['model_path'] = 'saved_models/cvae_saved_dict.pt'
CONFIG['scaler_path'] = 'saved_models/scaler.gz'
CONFIG['scaler_path2'] = 'saved_models/scaler2.gz'

CONFIG['train_model'] = True
CONFIG['generate_samps'] = 5

CONFIG['descriptor'] = ['ead', 'distance']

'''

CONFIG['data_x_path'] = 'MP_data_csv/perov_5_raw_train_dist_mat.pt'
CONFIG['data_ead_path'] = 'MP_data_csv/perov_5_raw_train_ead_mat.pt'
CONFIG['composition_path'] = 'MP_data_csv/perov_5_raw_train_composition_mat.pt'
CONFIG['cell_path'] = 'MP_data_csv/perov_5_raw_train_cell_mat.pt'
CONFIG['data_y_path'] = "MP_data_csv/perov_5/raw_train/targets.csv"

CONFIG['unprocessed_path'] = 'MP_data_csv/perov_5_raw_train_unprocessed.txt'

#CONFIG['model_path'] = 'saved_models/cvae_saved.pt'
CONFIG['model_path'] = 'saved_models/cvae_saved_dict.pt'
CONFIG['scaler_path'] = 'saved_models/scaler.gz'
CONFIG['scaler_path2'] = 'saved_models/scaler2.gz'

CONFIG['train_model'] = True
CONFIG['generate_samps'] = 5

CONFIG = dotdict(CONFIG)
CONFIG['descriptor'] = ['ead', 'distance']

# EAD params

CONFIG['L'] = 1
CONFIG['eta'] = [1, 20, 90]
CONFIG['Rs'] = np.arange(0, 10, step=0.2)
CONFIG['derivative'] = False
CONFIG['stress'] = False

CONFIG['all_neighbors'] = True
CONFIG['perturb'] = False
CONFIG['load_pos'] = False
CONFIG['cutoff'] = 20.0
CONFIG['offset_count'] = 3

CONFIG = dotdict(CONFIG)

trainer = Trainer(CONFIG)

model = trainer.model

print(CONFIG.model_path)
model.load_state_dict(torch.load(CONFIG.model_path, map_location=torch.device('cpu')))
model.eval()
scaler = joblib.load(CONFIG.scaler_path)
scaler2 = joblib.load(CONFIG.scaler_path2)

train_y_mean = torch.mean(trainer.y_train)
print(train_y_mean)

train_y_mean = torch.tensor(-1.5)
print(train_y_mean)


data_list = []
n_samples = 20

i = 1
m = 0

print('a')

while len(data_list) < n_samples :
    
    m+=1
    y = train_y_mean.view(-1,1)
    z = torch.randn(1, CONFIG.params.latent_dim).to(trainer.device)

    # zy = z # for unconditional
    zy = torch.cat((z, y), dim=1)
    #y = torch.tensor([[2.8400]])

    rev_x = model.decoder(zy)
    rev_x_need_scaling, composition_vec = rev_x[:, :CONFIG.params.input_dim-101], rev_x[:,-101:]
    rev_x_need_scaling = rev_x_need_scaling/5 

    rev_x_scaled = scaler.inverse_transform(rev_x_need_scaling.cpu().data.numpy())
    rev_x_scaled = torch.tensor(rev_x_scaled, dtype=torch.float, device=trainer.device)

    #composition_vec = scaler2.inverse_transform(composition_vec.cpu().data.numpy())
    #composition_vec = torch.tensor(composition_vec, dtype=torch.float, device=trainer.device)

    tota_atm = torch.round(5*composition_vec[:,100])
    composition_vec = composition_vec[:,0:100]
    
 
    #print(composition_vec*10)
    
    top_values, top_indices = composition_vec.topk(6)
    new_tensor2 = torch.zeros_like(composition_vec)
    new_tensor2[0][top_indices] = top_values


    composition_vec = (torch.round(new_tensor2[0])) 
    composition_vec[composition_vec < 0] = 0 # May change this value.
    sum_comp = torch.sum(composition_vec[0])

    composition_vec = (composition_vec/5)
    
    atomic_vec = torch.round((composition_vec) * tota_atm).int()
    atomic_vec = atomic_vec.flatten()

    #if torch.sum(atomic_vec) == 0:# or (torch.any(cell > 3)):
    #    continue

    atomic_numbers = []

    # find atomic number of all the elements with non-zero compostions.

    # atomic number will be as a tensor.
    for j in range(100):
        atomic_numbers.append(atomic_vec[j]*[str(j+1)])

    # make list of lists. [[] , [] , []]
    atomic_numbers = [item for sublist in atomic_numbers for item in sublist]
    
    if len(atomic_numbers)!=5: # or (tota_atm<5):
       continue

    # cell
    
    #print('dist',rev_x_scaled[0][590:607])
    #print(rev_x_scaled.shape) # torch.Size([1, 607]) 
    
    cell = rev_x_scaled[0, 601:607]
    #print(cell.shape) # torch.Size([6])
    print("cell1: ", cell)  # cell1:  tensor([3.8627, 4.6770, 6.7244, 1.5247, 1.7004, 1.5864])
    cell[3:6] = cell[3:6] * 180 / np.pi
    print("cell2: ", cell)  #cell2:  tensor([ 3.8627,  4.6770,  6.7244, 87.3581, 97.4244, 90.8916])
 
    
    print("========= ", i)
    print('dist',rev_x_scaled[0][600])
    print('composition_vec',composition_vec)
    print('atomic_vec',atomic_vec)
    print('tot_atm', tota_atm)
    print('atomic_num', atomic_numbers)
    print("cell1: ", cell)

    placeholder = Atoms(numbers=[1], positions=[[0,0,0]], cell=cell.detach().cpu().numpy())
    cell = placeholder.get_cell()
    
    # fianl data dictionary
    data = {}
    data['positions'] = []
    data['atomic_numbers'] = np.array(atomic_numbers, dtype="int")
    data['cell'] = torch.tensor(np.array(cell), dtype=torch.float, device=torch_device_select(CONFIG.gpu))
    data['representation'] = torch.unsqueeze(rev_x_scaled[0,:CONFIG.params.input_dim-101-6], 0).detach()


    i+=1
    data_list.append(data)
    print('done')
    

    
print('i', i )
print('m', m )    
ratio = (i/m) 

print('efficiency of sampling',ratio) # (0.2-0.45) ~ 20%-45%


print(len(data_list))
#print(data_list)
    
# save data_list
now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d_%H-%M-%S")

f = open("./pkl_files/" + now + ".pkl", 'wb')
pickle.dump(data_list, f)
f.close()

print(now)