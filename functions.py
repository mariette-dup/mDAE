#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:15:22 2022


"""


import numpy as np
from numpy import pi
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as td

from utils import *

from torch.utils.data import Dataset



def missing_data_MCAR(data, percent, nb_col, nb_rows): 
    
    #on donne patient, colonne 
    #calcul des pourcentage de NA 
    nb_cases_na = round(percent*nb_col*nb_rows)

    x = data.clone()

    ind= torch.tensor([[i, j] for i in range(nb_rows) for j in range(nb_col)], device=x.device)
    idx = torch.randperm(ind.size(0))
    ind_list_l = ind[idx]

    ind_list = ind_list_l[0:nb_cases_na]

    rows = ind_list[:,0]
    columns = ind_list[:,1]
    x[rows,columns]=0

    bloup = data.clone()

    bloup[rows,columns]=1000
    r_NA_block_or,c_NA_block_or = torch.where(bloup!=1000)
    list_no_na_col = torch.stack([r_NA_block_or,c_NA_block_or],axis=1)

    return x, nb_cases_na, rows, columns, list_no_na_col

def missing_data_MAR(data, percent, percent_var, nb_col, nb_rows):

    mask = MAR_mask(data, percent, percent_var)
    x = data.clone()
    x[mask.bool()] = 0

    bloup = data.clone()

    bloup[mask.bool()]=1000

    r_NA,c_NA = torch.where(bloup==1000)

    nb_cases_na = len(r_NA)

    r_NA_block_or,c_NA_block_or = torch.where(bloup!=1000)
    list_no_na_col = torch.stack([r_NA_block_or,c_NA_block_or],axis=1)

    return x,nb_cases_na, r_NA, c_NA, list_no_na_col

def missing_data_MNAR(data, percent, percent_var, nb_col, nb_rows, exclude_inputs):

    mask = MNAR_mask_logistic(data, percent, percent_var, exclude_inputs)
    x = data.clone()
    x[mask.bool()] = 0

    bloup = data.clone()

    bloup[mask.bool()]=1000

    r_NA,c_NA = torch.where(bloup==1000)

    nb_cases_na = len(r_NA)

    r_NA_block_or,c_NA_block_or = torch.where(bloup!=1000)
    list_no_na_col = torch.stack([r_NA_block_or,c_NA_block_or],axis=1)

    return x,nb_cases_na, r_NA, c_NA, list_no_na_col
 
def missing_data_blocks (data, percent, taille_block, nb_col, nb_rows):

    
    nb_rows_na = int((nb_rows*nb_col*percent)/taille_block)
    x = data.clone()
 

    ind= torch.tensor([[i, j] for i in range(nb_rows) for j in range((nb_col-taille_block)+1)], device=x.device)
    idx = torch.randperm(ind.size(0))
    ind_list_l = ind[idx]

    ind_list = ind_list_l[0:nb_rows_na]

    rows = ind_list[:,0]
    cols = ind_list[:,1].view(-1,1)

    elec = torch.arange(0,taille_block,device=x.device)
    cols = cols + elec
    cols = cols.view(-1).int()

    rows = torch.repeat_interleave(rows, taille_block)

   
    x[rows,cols] = 0
    
    bloup = data.clone()

    bloup[rows,cols]=1000


    nb_cases_na = len(rows)

    r_NA_block_or,c_NA_block_or = torch.where(bloup!=1000)
    list_no_na_col = torch.stack([r_NA_block_or,c_NA_block_or],axis=1)


    return x, nb_cases_na, rows, cols, list_no_na_col




def corruption_zeros_data_nodouble_valid2(data, percent, nb_col, nb_rows, ind_list): 
    
    #on donne patient, colonne 
    #calcul des pourcentage de NA 
    nb_cases_na = round(percent*len(ind_list))


    # np.random.shuffle(ind_list)
    idx = torch.randperm(ind_list.size(0))
    ind_list = ind_list[idx]
    

    ind_list = ind_list[0:nb_cases_na]


    x = data.clone()

    rows = ind_list[:,0]
    columns = ind_list[:,1]
    x[rows,columns]=0


    return x, nb_cases_na, rows, columns

class MyDataset(Dataset):
    def __init__(self,train_dataset, isna, transform=None, target_transform=None):
        self.data = train_dataset 
        self.isna = isna
        self.transform = transform 
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        data = self.data[index]
        isna = self.isna[index]
        
        # Your transformations here (or set it in CIFAR10)
        
        return data, index, isna

    def __len__(self):
        return len(self.data)


class AEovc(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=35
        )
        self.encoder_output_layer = nn.Linear(
            in_features=35, out_features=42
        )
        # self.s1 = nn.Linear(
        #     in_features=500, out_features=600
        # )
        # self.s3 = nn.Linear(
        #     in_features=170, out_features=220
        # )



        # self.s4 = nn.Linear(
        #     in_features=220, out_features=170
        # )
        # self.s2 = nn.Linear(
        #     in_features=600, out_features=500
        # )
        self.decoder_hidden_layer = nn.Linear(
            in_features=42, out_features=35
        )
        self.decoder_output_layer = nn.Linear(
            in_features=35, out_features=kwargs["input_shape"]
        )
        #self.dropout1 =nn.Dropout(p=0.2)

    def forward(self, features):
        
        #features = self.dropout1(features)
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)

        #activation = self.dropout1(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        
        #code = self.dropout1(code)
        # code = self.s1(code)
        # code = torch.relu(code)

        # code = self.dropout1(code)
        # code = self.s3(code)
        # code = torch.relu(code)

        # code = self.dropout1(code)
        # code = self.s4(code)
        # code = torch.relu(code)

        #code = self.dropout1(code)
        # code = self.s2(code)
        # code = torch.relu(code)

        #code = self.dropout1(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)

        #activation = self.dropout1(activation)
        activation = self.decoder_output_layer(activation)

        return activation

def miwae(z_withNA,x,h,d,K,bs,n_epochs):

    nb_col = z_withNA.size(1) 
    nb_rows = z_withNA.size(0)     

    mask = np.isfinite(z_withNA.cpu().numpy())
    

    decoder = nn.Sequential(
        torch.nn.Linear(d, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, 3*nb_col),  # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
    )
    
    encoder = nn.Sequential(
        torch.nn.Linear(nb_col, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, 2*d),  # the encoder will output both the mean and the diagonal covariance
    )     

    p_z = td.Independent(td.Normal(loc=torch.zeros(d).cpu(),scale=torch.ones(d).cpu()),1)

    def miwae_loss(iota_x,mask):
        batch_size = iota_x.shape[0]
        out_encoder = encoder(iota_x)
        q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)

        zgivenx = q_zgivenxobs.rsample([K])
        zgivenx_flat = zgivenx.reshape([K*batch_size,d])

        out_decoder = decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :nb_col]
        all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., nb_col:(2*nb_col)]) + 0.001
        all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*nb_col):(3*nb_col)]) + 3

        data_flat = torch.Tensor.repeat(iota_x,[K,1]).reshape([-1,1])
        tiledmask = torch.Tensor.repeat(mask,[K,1])

        all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,nb_col])

        logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size])
        logpz = p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))

        return neg_bound

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=1e-3)

    def miwae_impute(iota_x,mask,L):
        batch_size = iota_x.shape[0]
        out_encoder = encoder(iota_x)
        q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)
        
        zgivenx = q_zgivenxobs.rsample([L])
        zgivenx_flat = zgivenx.reshape([L*batch_size,d])
        
        out_decoder = decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :nb_col]
        all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., nb_col:(2*nb_col)]) + 0.001
        all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*nb_col):(3*nb_col)]) + 3
        
        data_flat = torch.Tensor.repeat(iota_x,[L,1]).reshape([-1,1]).cpu()
        tiledmask = torch.Tensor.repeat(mask,[L,1]).cpu()
        
        all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L*batch_size,nb_col])
        
        logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([L,batch_size])
        logpz = p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)
        
        xgivenz = td.Independent(td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model),1)

        imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
        xms = xgivenz.sample().reshape([L,batch_size,nb_col])
        xm=torch.einsum('ki,kij->ij', imp_weights, xms) 
        

        
        return xm
    
    def weights_init(layer):
        if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)

    miwae_loss_train=np.array([])
    mse_train=np.array([])
    mse_train2=np.array([])
    

    xhat_0 = x.cpu().numpy()

    xhat = np.copy(xhat_0) # This will be out imputed data matrix

    encoder.apply(weights_init)
    decoder.apply(weights_init)

    for ep in range(1,n_epochs):
        perm = np.random.permutation(nb_rows) # We use the "random reshuffling" version of SGD
        batches_data = np.array_split(xhat_0[perm,], nb_rows/bs)
        batches_mask = np.array_split(mask[perm,], nb_rows/bs)
        for it in range(len(batches_data)):
            optimizer.zero_grad()
            encoder.zero_grad()
            decoder.zero_grad()
            b_data = torch.from_numpy(batches_data[it]).float().cpu()
            b_mask = torch.from_numpy(batches_mask[it]).float().cpu()
            loss = miwae_loss(iota_x = b_data,mask = b_mask)
            loss.backward()
            optimizer.step()
        if ep % 100 == 1:
            print('Epoch %g' %ep)
            print('MIWAE likelihood bound  %g' %(-np.log(K)-miwae_loss(iota_x = torch.from_numpy(xhat_0).float().cpu(),mask = torch.from_numpy(mask).float().cpu()).cpu().data.numpy())) # Gradient step      
            
            ### Now we do the imputation
            
            xhat[~mask] = miwae_impute(iota_x = torch.from_numpy(xhat_0).float().cpu(),mask = torch.from_numpy(mask).float().cpu(),L=10).cpu().data.numpy()[~mask]
            
            # err = np.array([mse(xhat,xfull,mask)])
            # mse_train = np.append(mse_train,err,axis=0)
            # print('Imputation MSE  %g' %err)
            print('-----')

    return xhat 