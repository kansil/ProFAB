# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:46:12 2020

@author: Sameitos
"""

import numpy as np
import re
import csv
import json
# from numpyencoder import NumpyEncoder


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def get_reps(file_name):  
    '''

    Description: obtains representative protein in indicated unitref protein clusters

    Parameter
    ---------
    file_name: name of the file
    
    Return
    ------
    unitref_reps: Representatives of each protein_cluster in numpy array form

    '''  
    unitref_reps = []
    with open(file_name) as f:
        for rows in f:
            unitref_reps.append(rows.strip('\n\r'))
            
    return np.array(unitref_reps)

def map_ID(chembl_unitprot_mapping):
    '''

    Description: obtains both Chembl ID's and Unitprot IDs of protein in indicated file

    Parameter
    ---------
    chembl_unitprot_mapping: name of the file
    
    Return
    ------
    chembl_ID: chembl IDs of proteins
    prot_ids: unitprot IDs of proteins

    '''  
    chembl_ID = [] 
    prot_ids = []
    with open(chembl_unitprot_mapping) as f:
        for i,rows in enumerate(f):
            if i>0:
                rows = re.split('\t',rows)
                chembl_ID.append(rows[1])
                prot_ids.append(rows[0])

    chembl_ID = np.array(chembl_ID)
    prot_ids = np.array(prot_ids)

    return chembl_ID,prot_ids


def get_target(target_list,chembl_ID,prot_ids):
    '''

    Description: obtains both Chembl ID's and Unitprot IDs of protein in indicated file

    Parameter
    ---------
    chembl_unitprot_mapping: name of the file
    
    Return
    ------
    cluster_target: chembl ID + unitprot ID of target
    
    '''  
    map_indices = []
    for i in target_list:
        map_indices.append(int(np.where(chembl_ID == i)[0]))

    cluster_target = np.append(target_list.reshape(len(target_list),1),
                            np.take(prot_ids,map_indices).reshape(len(target_list),1),axis = 1)
    return cluster_target



def get_values(interaction_data):
    '''

    Description: obtains the indices of targets in bioactivity data by their number of non-nan values vs compounds

    Parameter
    ---------
    interaction_data: name of the file of bioactivity data
    
    Return
    ------
    crucials: indices of target that have non-nan values vs compounds more than 10 
    lesser: indices of target that have non-nan values vs compounds less than(or equal to) 10

    '''  
    
    interactions = []
    with open(interaction_data) as f:
        f = csv.reader(f,delimiter = ',')
        for rows in f:
            interactions.append(np.array(rows))
    idx_data = []
    for i in range(len(interactions[0])):
        data_number = []
        for j in range(len(interactions)):
            if interactions[j][i] !='nan': data_number.append(interactions[j][i])
        idx_data.append([len(data_number),i])
    interactions.clear()
    
    crucials,lessers = [],[]
    for i in idx_data:
        if i[0]>10: 
            crucials.append(i[1])
        else: lessers.append(i[1])
    return crucials,lessers


def form_cluster(crucials,lessers,cluster_target,unitref_reps):
    '''

    Description: obtains indices of targets in representatives

    Parameter
    ---------
    crucials: name of the file of bioactivity data
    lessers: indices of target that have non-nan values vs compounds less than(or equal to) 10
    cluster_target: a list of Uniprot IDs and Chembl IDs of targets
    unitref_reps: representavives

    Return
    ------
    high_reps: indices of target that have non-nan values vs compounds more than 10 in representatives
    less_reps: indices of target that have non-nan values vs compounds less than(or equal to) 10 in representatives
    higher_empty: indices of target that have non-nan values vs compounds more than 10 but not found in representatives

    '''  
    high_reps = []
    higher_empty = []
    for i in crucials:
        reps = np.where(unitref_reps == cluster_target[i,1])[0]
        
        if not reps:
            higher_empty.append(i)
        else: high_reps.append(i)
    less_reps = []
    for j in lessers:
        reps = np.where(unitref_reps == cluster_target[j,1])[0]
        if reps: less_reps.append([j,reps[0]])
    if len(less_reps)==0 or len(high_reps) ==0:
        return np.array(high_reps),np.array(less_reps),np.array(higher_empty)
    
    less_reps = np.array(less_reps)
    less_reps = less_reps[np.argsort(less_reps[:, 1])]
    
    return np.array(high_reps),less_reps,np.array(higher_empty)


def unitref_append(less_reps,file_name):
    '''

    Description: finds indices of target that has less than 10 non-nan values in bioactivity data, in unitref file 

    Parameter
    ---------
    less_reps: name of the file of bioactivity data
    file_name: file name of unitref file

    Return
    ------
    unitref_ID: values of unitref targets where elements of less_reps are indexed to

    '''  
    if len(less_reps)>0:
        unitref_ID = []
        with open(file_name) as f:
            for i,rows in enumerate(f):
                for k,j in enumerate(less_reps[:,1]):
                    if i == j:
                        rows =re.split('\t',rows.strip('n\r'))
                        rows_prot =re.split(';',rows[-1])
                        rows_prot[-1] =rows_prot[-1][:-1] 
                        unitref_ID.append(rows_prot)
                        if less_reps[k,1] == less_reps[-1,1]:
                            return unitref_ID
                        less_reps = less_reps[k+1:,:]
                        break
                    break
            return unitref_ID
    else:
        return []   

def high_in_low(higher_empty,less_reps,unitref_ID,cluster_target):
    '''

    Description: changes values of less_reps with higher_empty that is found in cluster of less_reps

    Parameter
    ---------
    higher_empty: indices of target that have non-nan values vs compounds more than 10 but not found in representatives
    less_reps: indices of target that have non-nan values vs compounds less than(or equal to) 10
    unitref_ID: values of unitref targets where elements of less_reps are indexed to
    cluster_target: cluster_target: chembl ID + unitprot ID of target

    Return
    ------ 
    less_reps: indices of target that have non-nan values vs compounds less than(or equal to) 10
 
    ''' 
    H_in_L = []
    for i in higher_empty:
        for j,idx in enumerate(less_reps):
            high_in_less = np.where(unitref_ID[j] == cluster_target[i,1])[0]
            if high_in_less:
                H_in_L.append([i,j])
    if not H_in_L:
        return less_reps,True
    else:
        for i,j in H_in_L:
            less_reps[j] = i
        return less_reps,True


def dumping(file_name,high_reps,higher_empty,less_reps,unitref_ID,cluster_target):
    '''

    Description: dumps the indices of targets that are found in unitref clusters in a json file 

    Parameter
    ---------
    file_name: file name of .json file 
    higher_empty: indices of target that have non-nan values vs compounds more than 10 but not found in representatives
    less_reps: indices of target that have non-nan values vs compounds less than(or equal to) 10
    unitref_ID: values of unitref targets where elements of less_reps are indexed to
    cluster_target: cluster_target: chembl ID + unitprot ID of target
    ---------

    '''

    less_reps,value = high_in_low(higher_empty,less_reps,unitref_ID,cluster_target)
    if value:
        with open(file_name,'w') as f: 
            if len(less_reps) == 0: 
                json.dump(high_reps.tolist(),f,cls=NumpyEncoder)
                return 
            clustered_proteins = np.unique(np.sort(np.append(high_reps,less_reps[:,0]))).tolist()
            json.dump(clustered_proteins,f,cls=NumpyEncoder)

