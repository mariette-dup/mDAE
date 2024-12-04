import torch
from functions import *
from functions_structure import * 




def mDAE (x, nb_col, nb_rows, nb_iter,prop_data_validation,list_no_na_col,ind_rows_na,ind_columns_na,epochs_mdae,batch_size,learning_rate, grid_mu, nb_cases_na,number_structure) :   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ##############  ARRAY FOR THE RESULTS #########  

    data_boxplot_mdae = torch.zeros([nb_cases_na])
    data_boxplot_true= torch.zeros([nb_cases_na])

    global_model = []

    nb_mu = len(grid_mu)
    tab_opti_mu = torch.zeros([nb_iter,nb_mu])

    v=0
    for j in grid_mu : 

        w=0 
        for d in range(nb_iter) : 
            
        
            z_eval, nb_cases_na, ind_rows_eval, ind_columns_eval = corruption_zeros_data_nodouble_valid2(x,prop_data_validation, nb_col, nb_rows,ind_list=list_no_na_col)
            
            zfordataloader = torch.zeros_like(x)
            zfordataloader[ind_rows_na,ind_columns_na]=1
            zfordataloader[ind_rows_eval,ind_columns_eval]=2
        
            epochs = epochs_mdae
            
            dataset = MyDataset(z_eval,zfordataloader)
            
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            
            model = choice(d_in=nb_col,number_structure=number_structure).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            loss_list = []
            model.train()

            
            for epoch in range(epochs):
                
                loss = 0
                for batch_features in train_loader:
                    
                    
                    batch_features,index, isna = batch_features
                    batch_features = batch_features.to(device)
                    index = index.to(device)
                    isna = isna.to(device)

                    r_na,c_na = torch.where(isna==0)

                    list_valid = torch.stack([r_na,c_na],axis=1)

                    batch_features_NA, nb_cases_na2, ind_rows_zeros, ind_columns_zeros = corruption_zeros_data_nodouble_valid2(batch_features,j, nb_col, len(batch_features),ind_list=list_valid)

                    r_vraie,c_vraie = torch.where(isna==0)
                    mask = torch.zeros([len(batch_features),nb_col],device=device)
                    mask[r_vraie,c_vraie]=1
                    optimizer.zero_grad()

                    outputs_strat1 = model(batch_features_NA)
                
                    
                    outputs_strat1 = outputs_strat1 *mask 
                    train_loss = criterion(outputs_strat1, batch_features)
            
                    train_loss.backward()
        

                    optimizer.step()
                    loss += train_loss.item()
                
                # compute the epoch training loss
                loss = loss / len(train_loader)
                loss_list.append(loss)

        
            pred_eval = model(z_eval)

            #test error 
            mask = torch.zeros([nb_rows,nb_col], device=device)
            mask[ind_rows_eval,ind_columns_eval]=1

            masked_z = x*mask
            masked_pred_eval = pred_eval*mask

            num_test_error = torch.sum((masked_z-masked_pred_eval)**2)
            den_test_error = torch.sum((masked_z)**2)

            test_error = num_test_error/den_test_error
            
            tab_opti_mu[w,v] = test_error 
            
            w=w+1 


        global_model.append(model)    
        v=v+1   


    mu_moyen = torch.mean(tab_opti_mu,0)

    mu_opti = grid_mu[np.argmin(mu_moyen.detach().cpu().numpy())]

    

    #now mu is chosen we can proceed at the imputation 
    epochs = epochs_mdae


    zfordataloader = torch.zeros_like(x)
    zfordataloader[ind_rows_na,ind_columns_na]=1


    dataset = MyDataset(x,zfordataloader)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = choice (d_in=nb_col,number_structure=number_structure).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    loss_list = []
    model.train()


    for epoch in range(epochs):
        
        loss = 0
        for batch_features in train_loader:
            
            batch_features,index, isna = batch_features
            batch_features = batch_features.to(device)
            index = index.to(device)
            isna = isna.to(device)

        
            r_na,c_na = torch.where(isna==0)

            list_valid = torch.stack([r_na,c_na],axis=1)

            batch_features_NA, nb_cases_na2, ind_rows_zeros, ind_columns_zeros = corruption_zeros_data_nodouble_valid2(batch_features,mu_opti, nb_col, len(batch_features),ind_list=list_valid)
            
            mask = torch.zeros([len(batch_features),nb_col],device=device)
            mask[r_na,c_na]=1
            optimizer.zero_grad()

            outputs_strat1 = model(batch_features_NA)
            
            
            outputs_strat1 = outputs_strat1 *mask 
            train_loss = criterion(outputs_strat1, batch_features)
        
            train_loss.backward()


            optimizer.step()
            loss += train_loss.item()
            
        # compute the epoch training loss
        loss = loss / len(train_loader)
        loss_list.append(loss)

 

    model.eval()
    pred_vrai = model(x) 

    pred_vrai = pred_vrai.to(device)

    data_boxplot_mdae[:] = pred_vrai[ind_rows_na,ind_columns_na]
    data_boxplot_mdae = data_boxplot_mdae.cpu().detach()

    x_imputed = x.clone()
    x_imputed[ind_rows_na,ind_columns_na] = pred_vrai[ind_rows_na,ind_columns_na]
    x_imputed = x_imputed.cpu().detach()
    
   
    torch.save(data_boxplot_mdae,f"NA_reconstructed_from_mDAE.pt")
    
    
    return x_imputed 