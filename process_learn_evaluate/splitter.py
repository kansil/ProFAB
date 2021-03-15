# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:17:42 2020

@author: Sameitos
"""

from sklearn.model_selection import train_test_split

def ttv_split(X,y,ratio = 0.2): 
    
    if type(ratio) == float:
        return train_test_split(X,y,test_size = ratio)
    elif type(ratio) == list:
        
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = ratio[0])
        X_train,X_validation,y_train,y_validation = train_test_split(X,y,test_size = ratio[1]/(1-ratio[0]))
        
        return X_train,X_test,X_validation,y_train,y_test,y_validation