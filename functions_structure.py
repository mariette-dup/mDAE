#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:15:22 2022

"""


import torch
import torch.nn as nn
from torchsummary import summary
#from utils import *



class AE_uc_struct1(nn.Module):
    def __init__(self,d_in, **kwargs,):
        super().__init__()
        self.d_in = d_in
        nunit = {"layer1" : self.d_in,"layer2" : int(self.d_in//4),"layer3" : int(self.d_in//16)} 
        
        self.encoder_hidden_layer = nn.Linear(
            in_features=nunit["layer1"], out_features=nunit["layer2"]
        )
        self.encoder_output_layer = nn.Linear(
            in_features=nunit["layer2"], out_features=nunit["layer3"]
        )
        
        self.decoder_hidden_layer = nn.Linear(
            in_features=nunit["layer3"], out_features=nunit["layer2"]
        )
        self.decoder_output_layer = nn.Linear(
            in_features=nunit["layer2"], out_features=nunit["layer1"]
        )
       

    def forward(self, features):
        
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)

       
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)

        activation = self.decoder_output_layer(activation)

        return activation
    

class AE_uc_struct2(nn.Module):
    def __init__(self,d_in, **kwargs,):
        super().__init__()
        self.d_in = d_in
        nunit = {"layer1" : self.d_in,"layer2" : int(self.d_in//2),"layer3" : int(self.d_in//4)} 
        
        self.encoder_hidden_layer = nn.Linear(
            in_features=nunit["layer1"], out_features=nunit["layer2"]
        )
        self.encoder_output_layer = nn.Linear(
            in_features=nunit["layer2"], out_features=nunit["layer3"]
        )
    

        self.decoder_hidden_layer = nn.Linear(
            in_features=nunit["layer3"], out_features=nunit["layer2"]
        )
        self.decoder_output_layer = nn.Linear(
            in_features=nunit["layer2"], out_features=nunit["layer1"]
        )
        
    def forward(self, features):
        
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)

        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        
        

        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)

        activation = self.decoder_output_layer(activation)

        return activation

class AE_oc_struct3(nn.Module):
    def __init__(self,d_in, **kwargs,):
        super().__init__()
        self.d_in = d_in
        nunit = {"layer1" : self.d_in,"layer2" : self.d_in*2,"layer3" : self.d_in*4} 
        
        self.encoder_hidden_layer = nn.Linear(
            in_features=nunit["layer1"], out_features=nunit["layer2"]
        )
        self.encoder_output_layer = nn.Linear(
            in_features=nunit["layer2"], out_features=nunit["layer3"]
        )
        
        self.decoder_hidden_layer = nn.Linear(
            in_features=nunit["layer3"], out_features=nunit["layer2"]
        )
        self.decoder_output_layer = nn.Linear(
            in_features=nunit["layer2"], out_features=nunit["layer1"]
        )
       

    def forward(self, features):
        
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)

       
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)

        activation = self.decoder_output_layer(activation)

        return activation

class AE_oc_struct4(nn.Module):
    def __init__(self,d_in, **kwargs,):
        super().__init__()
        self.d_in = d_in
        nunit = {"layer1" : self.d_in,"layer2" : self.d_in*2,"layer3" : self.d_in*4,"layer4" : self.d_in*8} 
        
        self.encoder_hidden_layer = nn.Linear(
            in_features=nunit["layer1"], out_features=nunit["layer2"]
        )
        self.encoder_output_layer = nn.Linear(
            in_features=nunit["layer2"], out_features=nunit["layer3"]
        )
        self.s1 = nn.Linear(
            in_features=nunit["layer3"], out_features=nunit["layer4"]
        )


        self.s2 = nn.Linear(
            in_features=nunit["layer4"], out_features=nunit["layer3"]
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=nunit["layer3"], out_features=nunit["layer2"]
        )
        self.decoder_output_layer = nn.Linear(
            in_features=nunit["layer2"], out_features=nunit["layer1"]
        )
       

    def forward(self, features):
        
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)

       
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)

        code = self.s1(code)
        code = torch.relu(code)


        code = self.s2(code)
        code = torch.relu(code)
        
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)

        activation = self.decoder_output_layer(activation)

        return activation
    
class AE_oc_struct5(nn.Module):
    def __init__(self,d_in, **kwargs,):
        super().__init__()
        self.d_in = d_in
        nunit = {"layer1" : self.d_in,"layer2" : self.d_in*2,"layer3" : self.d_in*4,"layer4" : self.d_in*8, "layer5" : self.d_in*16} 
        
        self.encoder_hidden_layer = nn.Linear(
            in_features=nunit["layer1"], out_features=nunit["layer2"]
        )
        self.encoder_output_layer = nn.Linear(
            in_features=nunit["layer2"], out_features=nunit["layer3"]
        )
        self.s1 = nn.Linear(
            in_features=nunit["layer3"], out_features=nunit["layer4"]
        )
        self.s3 = nn.Linear(
            in_features=nunit["layer4"], out_features=nunit["layer5"]
        )
        

        self.s4 = nn.Linear(
            in_features=nunit["layer5"], out_features=nunit["layer4"]
        )
        self.s2 = nn.Linear(
            in_features=nunit["layer4"], out_features=nunit["layer3"]
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=nunit["layer3"], out_features=nunit["layer2"]
        )
        self.decoder_output_layer = nn.Linear(
            in_features=nunit["layer2"], out_features=nunit["layer1"]
        )
        
    def forward(self, features):
        
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)

        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        
        code = self.s1(code)
        code = torch.relu(code)

        code = self.s3(code)
        code = torch.relu(code)


        code = self.s4(code)
        code = torch.relu(code)

        code = self.s2(code)
        code = torch.relu(code)

        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)

        activation = self.decoder_output_layer(activation)

        return activation

class AE_oc_struct6(nn.Module):
    def __init__(self,d_in, **kwargs,):
        super().__init__()
        self.d_in = d_in
        nunit = {"layer1" : self.d_in,"layer2" : self.d_in*5, "layer3" : self.d_in*25} 
        
        self.encoder_hidden_layer = nn.Linear(
            in_features=nunit["layer1"], out_features=nunit["layer2"]
        )
        self.encoder_output_layer = nn.Linear(
            in_features=nunit["layer2"], out_features=nunit["layer3"]
        )
        

        self.decoder_hidden_layer = nn.Linear(
            in_features=nunit["layer3"], out_features=nunit["layer2"]
        )
        self.decoder_output_layer = nn.Linear(
            in_features=nunit["layer2"], out_features=nunit["layer1"]
        )
        
    def forward(self, features):
        
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)

        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        
        

        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)

        activation = self.decoder_output_layer(activation)

        return activation
    


    

def choice (d_in,number_structure) : 

    if number_structure == 1 : 
        net = AE_uc_struct1(d_in)

    if number_structure == 2 : 
        net = AE_uc_struct2(d_in)
    
    if number_structure == 3 : 
        net = AE_oc_struct3(d_in)
    
    if number_structure == 4 : 
        net = AE_oc_struct4(d_in)
    
    if number_structure == 5 : 
        net = AE_oc_struct5(d_in)
    
    if number_structure == 6 : 
        net = AE_oc_struct6(d_in)


    return net
