# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:45:38 2020

@author: Sameitos
"""

import os
os.chdir(os.getcwd())
import numpy as np
import csv
import re
import json
from collections import defaultdict

def target_indices(dataset_name):
    feature_c = np.array([])
    feature_t = np.array([])
    
    if os.path.isfile(dataset_name + '/representatives.json'):
        with open(dataset_name + '/representatives.json') as f:
            content  = f.read()
            if  content:
                feature_c =np.sort(np.array(json.loads(content)))

    if os.path.isfile(dataset_name + '/cluster_protein50_idx.json'):
        with open(dataset_name + '/cluster_protein50_idx.json') as f:
            content  = f.read()
            if  content:
                feature_t =np.sort(np.array(json.loads(content)))    
    
    return feature_c,feature_t

def get_indices(dataset_name):
    
    """
    Parameters
    ----------
    dataset_name : name of dataset_name where dataset is stored
    Returns
    -------
    indices : indices of floats in the dataset
    bioactivity_data : compound target interacition data

    """

    bioactivity_data  = []
    with open('../proj_data' + dataset_name[2:] +'/bioactivity_data.csv') as sf:
        f = csv.reader(sf,delimiter = ',')
        for rows in f:
            bioactivity_data.append(rows)
    bioactivity_data = np.array(bioactivity_data)
    
    idx_c,idx_t = np.where(np.array(bioactivity_data) != 'nan')
    
    indices = np.append(idx_c.reshape(len(idx_t),1),idx_t.reshape(len(idx_t),1),axis = 1)
    indices = np.array(indices,dtype = 'int')
    bioactivity_data= np.array([bioactivity_data[i,j] for i,j in indices],dtype = 'float64')
    
    return indices,bioactivity_data

def get_features(dataset_name):
    
    target = defaultdict(list)
    with open(dataset_name + '/target_features.txt') as f:
        for rows in f:
            rows = re.split(' ',rows.strip('\n'))
 
            target[int(float(rows[0]))].append(rows[1:])
            
    compound = defaultdict(list)
    with open(dataset_name + '/compound_features.txt') as f:
        for rows in f:
            rows = re.split(' ',rows.strip('\n'))
            compound[int(float(rows[0]))].append(rows[1:])

    return compound,target

def forming_data(indices,y,compound,target):

    a = []
    for i,j in enumerate(indices):  
        try:
            a.append([j[0]] + [j[1]] + list(np.array(compound[j[0]][0],
                dtype = 'int')) + list(np.array(target[j[1]][0],
                dtype = 'float64')) +  [y[i]])
        except:
            pass

    return np.array(a)

def get_final_dataset(dataset_name):

    compound,target = get_features(dataset_name)

    indices,bioacitivity_data = get_indices(dataset_name)

    dataset = forming_data(indices,bioacitivity_data,compound,target)

    return dataset
