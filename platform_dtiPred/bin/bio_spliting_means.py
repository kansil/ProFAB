
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:17:42 2020

@author: Sameitos
"""
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from bio_loading_features_indices import target_indices


class splitter():
    
    def __init__(self,dataset_name,feature,ratio,split_type):  
        self.feature = feature
        self.ratio = ratio
        self.split_type = split_type
        self.dataset_name = dataset_name
    def random_split(self,dataset):   
        
        
        # if self.split_type !='random_split':
        #     with open(self.dataset_name + "/" + self.split_type + "_indices.txt",'w') as f:
        #         np.savetxt(f,dataset[:,:2],fmt='%d' )
        
        X_train,X_test,y_train,y_test = train_test_split(dataset[:,:-1],
                                                         dataset[:,-1],
                                                         test_size = self.ratio)
        X_train,X_validation,y_train,y_validation = train_test_split(X_train,
                                                               y_train,
                                                               test_size = self.ratio)
        
        return X_train,X_test,X_validation,y_train,y_test,y_validation
        
    def indexing(self,dataset,reps,c = True):   
        if c:return np.array([i for i in dataset if i[0] in reps])
        else:return np.array([i for i in dataset if i[1] in reps])
        
    def compound_split(self,dataset):
        
        feature = self.feature[0]
        dataset = self.indexing(dataset,feature)
        return  self.random_split(dataset)
    
    def target_split(self,dataset): 
        
        feature  = self.feature[1]
        dataset= self.indexing(dataset,feature,c=False)
        return  self.random_split(dataset)
    
    def target_compound_split(self,dataset):
        
        feature_c = self.feature[0]
        feature_t = self.feature[1]
                
        dataset = self.indexing(self.indexing(dataset,feature_c),feature_t,c=False)
        return  self.random_split(dataset)
    
def splitting(dataset_name,dataset,split_type,ratio):
    
    feature = target_indices(dataset_name)
    
    s = splitter(dataset_name,feature,ratio,split_type)
    split_methods = {'random_split':s.random_split,'compound_based':s.compound_split,
                       'target_based':s.target_split,
                       'target_compound_based':s.target_compound_split}
    
                    
    return split_methods[split_type](dataset)
    