# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:20:05 2020

@author: Sameitos
"""


        
class own_kfold():
    """
    splits the according to given indices.
    """
    def __init__(self,indices):
        self.indices = indices
    
    def get_n_splits(self,X,y = None,groups = None):
        
        return len(self.indices)
    
    def split(self,X,y=None,groups = None):
        
        for i in range(len(self.indices)): 
            test_indices = self.indices[i]
            train_indices = self.indices[i-3] + self.indices[i-2 ]+ self.indices[i-1]
            yield train_indices,test_indices 

