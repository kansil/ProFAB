# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:57:23 2020

@author: Sameitos
"""
import os
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

def load_data(file_name): 
    
    """
    This function is used to extract SMILES data of compounds in dataset_name.

    Parameters:
        file_name: It is folder name that holds list of compounds name
    Return:
        compound_list: list of compound name
    """   

    pPath = os.path.split(os.path.realpath(__file__))[0]
    smiles_data = {}
    with open (pPath + '/chembl27_chemreps.txt') as f:
        for row in f:
            row = re.split('\t',row.strip('\n'))
            try:
                smiles_data[row[0]].append(row[1])
            except:
                smiles_data.update({row[0]:row[1]})
                
    compound_list = []
    with open(file_name) as f:
        for k,row in enumerate(f):
            compound_list.append([row[:-1],k])   

    return compound_list,smiles_data


def find_idx(IDs,smiles_data,bits,save_data,save_idx):
    
    """
    This function get the rdkit.BitVector from SMILES data of molecules.
    Parameters
    ----------
    IDs : chemble ID's of each compound
    smiles_data : smiles of each compound in chembl_27_chemreps.txt file
    bits: length of bitvectors
    save_data : file where fingerprints are stored
    save_idx = file where indices of data points are stored. It is beacuse some 
                data points can be lost during process.
    """
    non_convert = []
    sf =  open (save_data, 'w')
    idxf = open(save_idx, 'w')
    for i in IDs:
        try:
            row = smiles_data[i[0]]
            row = Chem.MolFromSmiles(row)
            row = AllChem.GetMorganFingerprintAsBitVect(row,2,nBits = bits)
            row = DataStructs.cDataStructs.BitVectToText(row)
            if save_data[-3:] == 'tsv':
                sf.write('%s\n' % '\t'.join([str(i[0]),''.join(row)]))
            elif save_data[-3:] == 'csv':
                sf.write('%s\n' % ','.join([str(i[0]),''.join(row)]))
            elif save_data[-3:] == 'txt':
                sf.write('%s\n' % ' '.join([str(i[0]),''.join(row)]))
            idxf.write('%s\n' % str(i[1]))
        except:
            non_convert.append(i)
    sf.close()
    idxf.close()
    return non_convert
