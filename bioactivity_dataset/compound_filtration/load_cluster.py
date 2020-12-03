# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:17:11 2020

@author: Sameitos
"""

import numpy as np
import os
import csv
import json

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from tqdm import tqdm

def get_data(dataset_name):
    """
    
    Parameters
    ----------
    dataset_name : name of the dataset_name where data obtained
    Returns
    -------
    dataset : a matrix consists of data chembl_IDs and their original indices
    fps : a matrix of bitvector

    """
    dataset = []
    fps = []
    with open ('../' + dataset_name + '/compound_features.txt', 'w') as sf:
        with open('../' +dataset_name +'/Compound_Smiles_Idx.txt') as f:
            s = csv.reader(f,delimiter = ' ')
            for k,rows in enumerate(tqdm(s)):
                rows[-1] = int(rows[-1])
                rows[1] = Chem.MolFromSmiles(rows[1])
                if rows[1] != None:
                    rows[1] = AllChem.GetMorganFingerprintAsBitVect(rows[1],2,1024)
                    dataset.append([rows[0],rows[2]])
                    fps.append(rows[1])
                    np.savetxt(sf,np.array([np.append(rows[2],np.array(rows[1],dtype = 'str'))]),fmt = '%s')
        return dataset,fps 


def simClustering(data,cutoff = 0.3):
    """
    
    Parameters
    ----------
    fps : Data, morgan fingerprints of compounds 

    Returns
    -------
    clstr : tuple, contains indices of points clustered via ButinaCluster of rdkit module
    
    """
    dist = []
    n_points = len(data)
    for i in tqdm(range(1,n_points)):
        sims = DataStructs.BulkTanimotoSimilarity(data[i],data[:i])
        dist.extend([1-x for x in sims])

    clstr = Butina.ClusterData(dist,n_points,cutoff,isDistData=True)
    return clstr

def c_indexing_loading(dataset_name,cluster_result, dataset):
    """

    Parameters
    ----------
    cluster_result : tuple,  contains indices of points clustered via ButinaCluster of rdkit module
    dataset : dataset contains compound name, their indices, molfingerprint and smiles data
    -------
    
    """
    cluster_indices = []
    for i,idx in enumerate(cluster_result):
        ci = []
        for j,idxj in enumerate(idx):
            ci.append(int(dataset[idxj,1]))
        cluster_indices.append(ci)
        
    representatives = []
    for i in cluster_indices:
        representatives.append(i[0])
    
    with open('../' + dataset_name + '/representatives.json','w') as f: 
        json.dump(representatives,f)
    
