# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:05:11 2023

@author: nisar
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd

def background_clutter_estimation (input_path,input_data):
    
# Load data
    #read all csv files in path
    #path_background_subtract="C://Users//nisar//Documents//immercom_session1_session2//data_mmwave_immercom//session2//background subtract//"
    path_background_subtract=input_path
    files_bg=[f for f in os.listdir(path_background_subtract) if f.endswith('.csv')]
    # Sort the file indices based on their modification time
    
    # Sort the files based on the sorted indices
    files=files_bg
    combined_bg=[]
    data=[]
    bg=[]
    for file in files:
        print(file)
        df=pd.read_csv(path_background_subtract+"//"+file,delimiter=",")
        df=df.dropna()
        #print(label)
       # print(df)
        x=df.iloc[:,0:30].to_numpy()
        #print(x.shape)
       # print(x)
        data.append(x)
        #print(files.index(file))
    
    
    bg=data
    for i in range(len(bg)):
        #print(i)
        if len(bg[i])%50==0:
            a=bg[i]
            n=len(bg[i])//50
        else:
            number_to_trunc=len(bg[i])//50
            length_trunc=50*number_to_trunc
            a=bg[i][:length_trunc,:]
            n=len(a)//50
        p=a.reshape(n,30,50)
        combined_bg.append(p)
        
        
    concat_bg=np.concatenate(combined_bg) 
    background_clutter=np.mean(concat_bg,axis=0)
    result=input_data.numpy()-background_clutter
    return background_clutter,torch.from_numpy(result)
# result=background_clutter_estimation("C://Users//nisar//Documents//immercom_session1_session2//data_mmwave_immercom//session2//background subtract//", torch.randn(50,30,50))