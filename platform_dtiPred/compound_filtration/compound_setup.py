<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 22:55:04 2020

@author: Sameitos
"""
import os 
from load_cluster import *
from get_smiles import load_data

def form_features(dataset_name, cluster = False):    
    
    if not os.path.exists('../' + dataset_name):
        os.makedirs('../' +dataset_name)
    
    """
    Parameter:
        cluster: It decides that whether function makes any clustering algoritm on data or not
                 If it stays False, create only a file 'compound_features.txt' otherwise it will create 
                 a file 'representative.txt' too. 
    """
    
    load_data(dataset_name)
    dataset,fps = get_data(dataset_name)
    
    if cluster:
        cluster_result = simClustering(fps)
        c_indexing_loading(dataset_name,cluster_result, np.array(dataset))


