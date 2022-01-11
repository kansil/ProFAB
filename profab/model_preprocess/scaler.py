# -*- coding: utf-8 -*-
"""
Created on Fri May 22 19:01:23 2020

@author: Sameitos
"""

import numpy as np

class ScalerNormalizer():
    """
    Description:This class has 5 methods in which normalizer stands to 
                           normalize the data. Others are for scaling
    
    
    Parameters
        train: train feature matrix
        
    Return
        scaled_train: feature matrix which function was applied
        scaler: a model to scale other dataset like test datatest
        
    """
    def normalizer(self,train,norm = 'l2'):
        #normalizer for the data
        from sklearn.preprocessing import Normalizer
        
        scaler = Normalizer(norm = norm)
        scaled_train = scaler.fit_transform(train)
        return scaled_train,scaler
  

      
    def standard_scaler(self,train):
        #Standard scaler or z-score scaler on any data 
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train)
        return scaled_train,scaler
        

        
    def maxabs_scaler(self,train):
        #for the sparse data
        from sklearn.preprocessing import MaxAbsScaler
        
        scaler = MaxAbsScaler()
        scaled_train = scaler.fit_transform(train)
        return scaled_train,scaler
        

    
    def minmax_scaler(self,train):
        #for the data has small std and is not gaussian dist
                      
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train)
        return scaled_train,scaler 
        

    
    def robust_scaler(self,train):
        #for the data to eleminate outliers
        from sklearn.preprocessing import RobustScaler
        
        scaler = RobustScaler()
        scaled_train = scaler.fit_transform(train)
        return scaled_train,scaler 
        

def scale_methods(X_train,scale_type = 'standard'):
    
    '''
    Description:
        This function is to run scale module of ProFAB

    Paramters:
        -**X_train**: type = {list, numpy array}, A data to train scaling functions
        -**scale_type**: {'normalizer','standard','max_abs','min_max','robust'}, 
                        default = 'standard, determines the method to scale the data.
    Return:
        X_train: type = {list, numpy array}, Transformed data.
        scaler: A fitting function to transform other datasets.
    '''


    s = ScalerNormalizer()

    scaler_ways = {'normalizer':s.normalizer,'standard':s.standard_scaler,
                   'max_abs':s.maxabs_scaler,'min_max': s.minmax_scaler,
                   'robust': s.robust_scaler}
    
    X_train,scaler = scaler_ways[scale_type](X_train)
    
    return X_train,scaler
