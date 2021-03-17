# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 22:55:04 2020

@author: Sameitos
"""
import os 
from smiles_to_bit import load_data,find_idx

def mol_to_bit(data_name,
 	save_data = 'bitvector.tsv',
    save_idx = 'indices.txt',
 	bits = 1024):
    '''
 	It is main file to convert SMILES data to Morgan fingerprints

 	Parameters:
		data_name: Name of file of SMILES data
		save_data: Name of file where fingerprits are stored. Its format can be .csv, .txt and .tsv
		save_idx: Name of file where indices of data points are stored. It is beacuse some 
                  data points can be lost during process.
        bits: number of dimensions of fingerprints
    '''
    if not os.path.exists(data_name):
        print(f'File {data_name} does not exist')
    IDs,smiles_data = load_data(data_name)
    
    find_idx(IDs = IDs,
              smiles_data = smiles_data,
              bits = bits,
              save_data = save_data,
              save_idx = save_idx)
