# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 21:16:13 2020

@author: Sameitos
"""
import numpy as np
import re

def push_features(dataset_name):
    
    # ID_features - np.loadtxt(dataset_name + '/PAAC.txt',dtype = 'str')
    # indices - np.loadtxt(dataset_name) + 
    """
    
    Description: sticks all protein features with their indexes which they are proteins 
    original place in raw dataset

    dataset_name: name of the file where file will be stored
   
    """

    ID_features = np.loadtxt('../' + dataset_name + '/PAAC.tsv',dtype = 'str')
    indices = np.loadtxt('../' + dataset_name + '/target_indices.txt', dtype = 'int')
    
    feature = np.array(ID_features[:,1:],dtype = 'float')
    
    np.savetxt('../' + dataset_name + '/target_features.txt', np.append(indices.reshape(len(indices),1),feature,axis = 1),fmt = '%s')
    
dataset_name = 'gpcr_data'
push_features(dataset_name)









