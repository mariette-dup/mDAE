import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import *


from functions import *
from functions_structure import * 
from utils import *





def sensibility_struct(x, grid_mu, nb_iter,nb_rows,nb_col,nb_cases_na,ind_rows_na,ind_columns_na,list_no_na_col,prop_data_validation, epochs_mdae,batch_size,learning_rate) : 

    nb_mu = len(grid_mu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    data_boxplot_mdae_struct1 = torch.zeros([nb_cases_na])
    data_boxplot_mdae_struct2 = torch.zeros([nb_cases_na])
    data_boxplot_mdae_struct3 = torch.zeros([nb_cases_na])
    data_boxplot_mdae_struct4 = torch.zeros([nb_cases_na])
    data_boxplot_mdae_struct5 = torch.zeros([nb_cases_na])
    data_boxplot_mdae_struct6 = torch.zeros([nb_cases_na])
    
    
    


    # ########### mDAE ########
    
    print("mdae struct1 ")
    

    tab_opti_mu = torch.zeros([nb_iter,nb_mu])

    v=0 
    global_model = []


    for j in grid_mu : 

        #start_time = time.time()

        w=0 
        for d in range(nb_iter) : 
            
        
            z_eval, nb_cases_na, ind_rows_eval, ind_columns_eval = corruption_zeros_data_nodouble_valid2(x,prop_data_validation, nb_col, nb_rows,ind_list=list_no_na_col)
            
            zfordataloader = torch.zeros_like(x)
            zfordataloader[ind_rows_na,ind_columns_na]=1
            zfordataloader[ind_rows_eval,ind_columns_eval]=2
        
            
            epochs = epochs_mdae
            
            
            
            dataset = MyDataset(z_eval,zfordataloader)
            
        
            
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            

            model = AE_uc_struct1(d_in=nb_col).to(device)

            
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
    
                #6 sec pour faire un batch avec e-2
                
                # display the epoch training loss
                # if np.remainder(epoch,200)==0 : 
                #     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 
                #2min pour faire 50 epochs 
        
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
        #break
        
        
        v=v+1


    #torch.save(tab_opti_mu,f"mdae_aleat_{percent_missing_data}.pt")

    mu_moyen = torch.mean(tab_opti_mu,0)

    mu_opti = grid_mu[np.argmin(mu_moyen.detach().cpu().numpy())]




    #impute
    epochs = epochs_mdae


    zfordataloader = torch.zeros_like(x)
    zfordataloader[ind_rows_na,ind_columns_na]=1


    dataset = MyDataset(x,zfordataloader)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AE_uc_struct1(d_in=nb_col).to(device)
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

        #6 sec pour faire un batch avec e-2
        
        # display the epoch training loss
        # if np.remainder(epoch,200)==0 : 
        #     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 

    # plt.figure()
    # plt.clf()
    # plt.plot(loss_list)
    # plt.xlabel("epochs")

    model.eval()
    pred_vrai = model(x) 

    pred_vrai = pred_vrai.to(device)

    data_boxplot_mdae_struct1[:] = pred_vrai[ind_rows_na,ind_columns_na]
    data_boxplot_mdae_struct1 = data_boxplot_mdae_struct1.cpu().detach()
    
    # ########### mDAE STRUCT 2 ########
    
    print("mdae struct2 ")
    


    tab_opti_mu = torch.zeros([nb_iter,nb_mu])

    v=0 
    global_model = []


    for j in grid_mu : 

        #start_time = time.time()

        w=0 
        for d in range(nb_iter) : 
            
        
            z_eval, nb_cases_na, ind_rows_eval, ind_columns_eval = corruption_zeros_data_nodouble_valid2(x,prop_data_validation, nb_col, nb_rows,ind_list=list_no_na_col)
            
            zfordataloader = torch.zeros_like(x)
            zfordataloader[ind_rows_na,ind_columns_na]=1
            zfordataloader[ind_rows_eval,ind_columns_eval]=2
        
            
            epochs = epochs_mdae
            
            
            
            dataset = MyDataset(z_eval,zfordataloader)
            
            
            
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            

            model = AE_uc_struct2(d_in=nb_col).to(device)
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
    
                #6 sec pour faire un batch avec e-2
                
                # display the epoch training loss
                # if np.remainder(epoch,200)==0 : 
                #     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 
                #2min pour faire 50 epochs 
        
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
        #break
        
        
        v=v+1


    #torch.save(tab_opti_mu,f"mdae_aleat_{percent_missing_data}.pt")

    mu_moyen = torch.mean(tab_opti_mu,0)

    mu_opti = grid_mu[np.argmin(mu_moyen.detach().cpu().numpy())]




    #impute
    epochs = epochs_mdae


    zfordataloader = torch.zeros_like(x)
    zfordataloader[ind_rows_na,ind_columns_na]=1


    dataset = MyDataset(x,zfordataloader)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AE_uc_struct2(d_in=nb_col).to(device)
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

        #6 sec pour faire un batch avec e-2
        
        # display the epoch training loss
        # if np.remainder(epoch,200)==0 : 
        #     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 

    # plt.figure()
    # plt.clf()
    # plt.plot(loss_list)
    # plt.xlabel("epochs")

    model.eval()
    pred_vrai = model(x) 

    pred_vrai = pred_vrai.to(device)

    data_boxplot_mdae_struct2[:] = pred_vrai[ind_rows_na,ind_columns_na]
    data_boxplot_mdae_struct2 = data_boxplot_mdae_struct2.cpu().detach()
    
    # ########### mDAE ########
    
    print("mdae struct3 ")
    
    

    tab_opti_mu = torch.zeros([nb_iter,nb_mu])

    v=0 
    global_model = []


    for j in grid_mu : 

        #start_time = time.time()

        w=0 
        for d in range(nb_iter) : 
            
        
            z_eval, nb_cases_na, ind_rows_eval, ind_columns_eval = corruption_zeros_data_nodouble_valid2(x,prop_data_validation, nb_col, nb_rows,ind_list=list_no_na_col)
            
            zfordataloader = torch.zeros_like(x)
            zfordataloader[ind_rows_na,ind_columns_na]=1
            zfordataloader[ind_rows_eval,ind_columns_eval]=2
        
            
            epochs = epochs_mdae
            
            
            
            dataset = MyDataset(z_eval,zfordataloader)
            
            
            
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            

            model = AE_oc_struct3(d_in=nb_col).to(device)
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
    
                #6 sec pour faire un batch avec e-2
                
                # display the epoch training loss
                # if np.remainder(epoch,200)==0 : 
                #     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 
                #2min pour faire 50 epochs 
        
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
        #break
        
        
        v=v+1


    #torch.save(tab_opti_mu,f"mdae_aleat_{percent_missing_data}.pt")

    mu_moyen = torch.mean(tab_opti_mu,0)

    mu_opti = grid_mu[np.argmin(mu_moyen.detach().cpu().numpy())]

    



    #impute
    epochs = epochs_mdae


    zfordataloader = torch.zeros_like(x)
    zfordataloader[ind_rows_na,ind_columns_na]=1


    dataset = MyDataset(x,zfordataloader)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AE_oc_struct3(d_in=nb_col).to(device)
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

        #6 sec pour faire un batch avec e-2
        
        # display the epoch training loss
        # if np.remainder(epoch,200)==0 : 
        #     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 

    # plt.figure()
    # plt.clf()
    # plt.plot(loss_list)
    # plt.xlabel("epochs")

    model.eval()
    pred_vrai = model(x) 

    pred_vrai = pred_vrai.to(device)

    data_boxplot_mdae_struct3[:] = pred_vrai[ind_rows_na,ind_columns_na]
    data_boxplot_mdae_struct3 = data_boxplot_mdae_struct3.cpu().detach()
    
    
    # ########### mDAE ########
    
    print("mdae struct4 ")
    
    

    tab_opti_mu = torch.zeros([nb_iter,nb_mu])

    v=0 
    global_model = []


    for j in grid_mu : 

        #start_time = time.time()

        w=0 
        for d in range(nb_iter) : 
            
        
            z_eval, nb_cases_na, ind_rows_eval, ind_columns_eval = corruption_zeros_data_nodouble_valid2(x,prop_data_validation, nb_col, nb_rows,ind_list=list_no_na_col)
            
            zfordataloader = torch.zeros_like(x)
            zfordataloader[ind_rows_na,ind_columns_na]=1
            zfordataloader[ind_rows_eval,ind_columns_eval]=2
        
            
            epochs = epochs_mdae
            
            
            
            dataset = MyDataset(z_eval,zfordataloader)
            
            
            
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            

            model = AE_oc_struct4(d_in=nb_col).to(device)
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
    
                #6 sec pour faire un batch avec e-2
                
                # display the epoch training loss
                # if np.remainder(epoch,200)==0 : 
                #     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 
                #2min pour faire 50 epochs 
        
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
        #break
        
        
        v=v+1


    #torch.save(tab_opti_mu,f"mdae_aleat_{percent_missing_data}.pt")

    mu_moyen = torch.mean(tab_opti_mu,0)

    mu_opti = grid_mu[np.argmin(mu_moyen.detach().cpu().numpy())]

    



    #impute
    epochs = epochs_mdae


    zfordataloader = torch.zeros_like(x)
    zfordataloader[ind_rows_na,ind_columns_na]=1


    dataset = MyDataset(x,zfordataloader)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AE_oc_struct4(d_in=nb_col).to(device)
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

        #6 sec pour faire un batch avec e-2
        
        # display the epoch training loss
        # if np.remainder(epoch,200)==0 : 
        #     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 

    # plt.figure()
    # plt.clf()
    # plt.plot(loss_list)
    # plt.xlabel("epochs")

    model.eval()
    pred_vrai = model(x) 

    pred_vrai = pred_vrai.to(device)

    data_boxplot_mdae_struct4[:] = pred_vrai[ind_rows_na,ind_columns_na]
    data_boxplot_mdae_struct4 = data_boxplot_mdae_struct4.cpu().detach()
    
    # ########### mDAE ########
    
    print("mdae struct5 ")
    
    

    tab_opti_mu = torch.zeros([nb_iter,nb_mu])

    v=0 
    global_model = []


    for j in grid_mu : 

        #start_time = time.time()

        w=0 
        for d in range(nb_iter) : 
            
        
            z_eval, nb_cases_na, ind_rows_eval, ind_columns_eval = corruption_zeros_data_nodouble_valid2(x,prop_data_validation, nb_col, nb_rows,ind_list=list_no_na_col)
            
            zfordataloader = torch.zeros_like(x)
            zfordataloader[ind_rows_na,ind_columns_na]=1
            zfordataloader[ind_rows_eval,ind_columns_eval]=2
        
            
            epochs = epochs_mdae
            
            
            
            dataset = MyDataset(z_eval,zfordataloader)
            
            
            
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            

            model = AE_oc_struct5(d_in=nb_col).to(device)
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
    
                #6 sec pour faire un batch avec e-2
                
                # display the epoch training loss
                # if np.remainder(epoch,200)==0 : 
                #     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 
                #2min pour faire 50 epochs 
        
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
        #break
        
        
        v=v+1


    #torch.save(tab_opti_mu,f"mdae_aleat_{percent_missing_data}.pt")

    mu_moyen = torch.mean(tab_opti_mu,0)

    mu_opti = grid_mu[np.argmin(mu_moyen.detach().cpu().numpy())]




    #impute
    epochs = epochs_mdae


    zfordataloader = torch.zeros_like(x)
    zfordataloader[ind_rows_na,ind_columns_na]=1


    dataset = MyDataset(x,zfordataloader)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AE_oc_struct5(d_in=nb_col).to(device)
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

        #6 sec pour faire un batch avec e-2
        
        # display the epoch training loss
        # if np.remainder(epoch,200)==0 : 
        #     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 

    # plt.figure()
    # plt.clf()
    # plt.plot(loss_list)
    # plt.xlabel("epochs")

    model.eval()
    pred_vrai = model(x) 

    pred_vrai = pred_vrai.to(device)

    data_boxplot_mdae_struct5[:] = pred_vrai[ind_rows_na,ind_columns_na]
    data_boxplot_mdae_struct5 = data_boxplot_mdae_struct5.cpu().detach()

# ########### mDAE ########
    
    print("mdae struct6 ")
    
    

    tab_opti_mu = torch.zeros([nb_iter,nb_mu])

    v=0 
    global_model = []


    for j in grid_mu : 

        #start_time = time.time()

        w=0 
        for d in range(nb_iter) : 
            
        
            z_eval, nb_cases_na, ind_rows_eval, ind_columns_eval = corruption_zeros_data_nodouble_valid2(x,prop_data_validation, nb_col, nb_rows,ind_list=list_no_na_col)
            
            zfordataloader = torch.zeros_like(x)
            zfordataloader[ind_rows_na,ind_columns_na]=1
            zfordataloader[ind_rows_eval,ind_columns_eval]=2
        
            
            epochs = epochs_mdae
            
            
            
            dataset = MyDataset(z_eval,zfordataloader)
            
            
            
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            

            model = AE_oc_struct6(d_in=nb_col).to(device)
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
    
                #6 sec pour faire un batch avec e-2
                
                # display the epoch training loss
                # if np.remainder(epoch,200)==0 : 
                #     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 
                #2min pour faire 50 epochs 
        
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
        #break
        
        
        v=v+1


    #torch.save(tab_opti_mu,f"mdae_aleat_{percent_missing_data}.pt")

    mu_moyen = torch.mean(tab_opti_mu,0)

    mu_opti = grid_mu[np.argmin(mu_moyen.detach().cpu().numpy())]

    

    #impute
    epochs = epochs_mdae


    zfordataloader = torch.zeros_like(x)
    zfordataloader[ind_rows_na,ind_columns_na]=1


    dataset = MyDataset(x,zfordataloader)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AE_oc_struct6(d_in=nb_col).to(device)
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

        #6 sec pour faire un batch avec e-2
        
        # display the epoch training loss
        # if np.remainder(epoch,200)==0 : 
        #     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 

    # plt.figure()
    # plt.clf()
    # plt.plot(loss_list)
    # plt.xlabel("epochs")

    model.eval()
    pred_vrai = model(x) 

    pred_vrai = pred_vrai.to(device)

    data_boxplot_mdae_struct6[:] = pred_vrai[ind_rows_na,ind_columns_na]
    data_boxplot_mdae_struct6 = data_boxplot_mdae_struct6.cpu().detach()


    torch.save(data_boxplot_mdae_struct1,f"data_boxplot_mdae_struct1.pt")
    torch.save(data_boxplot_mdae_struct2,f"data_boxplot_mdae_struct2.pt")
    torch.save(data_boxplot_mdae_struct3,f"data_boxplot_mdae_struct3.pt")
    torch.save(data_boxplot_mdae_struct4,f"data_boxplot_mdae_struct4.pt")
    torch.save(data_boxplot_mdae_struct5,f"data_boxplot_mdae_struct5.pt")
    torch.save(data_boxplot_mdae_struct6,f"data_boxplot_mdae_struct6.pt")


