# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:15:30 2020

@author: Sameitos
"""


from protein_selection import *

def get_target_indices(dataset_name,similarity):
    '''
    
    Parameters
    ----------
    dataset_name : folder name where data is stored
    similarity : similarity between clustered proteins
    -------

    '''
    file_unitref_reps = '../../uniref_protein/uniref_reps'+ str(similarity) + '.txt'
    unitref_reps = get_reps(file_unitref_reps)
    
    chembl_unitprot_mapping = 'chembl_uniprot_mapping.txt'
    chembl_ID,prot_ids = map_ID(chembl_unitprot_mapping)
    
    target_list = np.loadtxt('../proj_data/'+dataset_name+'/targets_ids_list.txt',dtype = 'str')
    cluster_target = get_target(target_list,chembl_ID,prot_ids)
    
    interaction_data = '../proj_data/' + dataset_name + '/bioactivity_data.csv'
    crucials,lessers = get_values(interaction_data)
    high_reps,less_reps,higher_empty = form_cluster(crucials,lessers,cluster_target,unitref_reps)
    
    file_unitref  = 'uniref_protein/uniref-filtered-identity' + str(similarity/100) + '.tab' 
    unitref_ID = unitref_append(less_reps,file_unitref)
    
    file_name = '../' +dataset_name + '/cluster_protein_idx.json' 
    dumping(file_name,high_reps,higher_empty,less_reps,unitref_ID,cluster_target)

 
dataset_name = 'gpcr_data'
similarity = 50
get_target_indices(dataset_name,similarity)
