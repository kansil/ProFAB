# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:57:23 2020

@author: Sameitos
"""

import numpy as np
import re
from collections import defaultdict
from tqdm import tqdm

def find_idx(IDs,smiles_data,f):
    """
    
    Parameters
    ----------
    IDs : chemble ID's of each compound
    smiles_data : smiles of each compound in chembl_27_chemreps.txt file
    f : file where smniles data are stored for each specific compound


    """
    for i in tqdm(range(len(IDs))):
        try:
            rows = smiles_data[IDs[i]][0]
            #rows = np.where(smiles_data[:,0] == IDs[i])[0] 
            lines = np.array([[IDs[i],rows,i]])
            np.savetxt(f,lines,fmt = '%s')
            
        except:
            # print('No smiles data is found') 
            pass

def load_data(dataset_name):         
    whole = defaultdict(list)
    with open ('whole.txt') as f:
        for rows in f:
            rows = re.split(' ',rows.strip('n\r'))
            whole[rows[0]].append(rows[1][:-1])
    
    
    #C:\Users\Sameitos\Desktop\proj_data\nr_data
    file_name = '../proj_data/' +dataset_name + '/compound_ids_list.txt'
    compound_list_ = np.loadtxt(file_name,dtype = 'str')
    
    with open('../' + dataset_name + '/Compound_Smiles_Idx.txt','w') as f:
        find_idx(compound_list_,whole,f)
    
