# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:26:00 2022

@author: Sameitos
"""

from ..utils import separator
import numpy as np
import random

def ttv_split(X = None,y = None, ratio = 0.2, X_pos = None, X_neg = None):
    
    '''
    Description:
        This function splits X and y randomly to train, test and validation
        sets according to ratio value
        
    Parameters:
        X: default = None, feature matrix. If X_pos and X_neg are not None,
            it has to stay None.
        y: default = None, label Matrix. If X_pos and X_neg are not None,
            it has to stay None. If X is defined, it cannot stay None.
        X_pos: default = None, positive set feature matrix. If X is defined,
                it has to stay None
        X_neg: default = None, negative set feature matrix. If X is defined,
                it has to stay None.
        ratio, type = {float,list}, (default = 0.2), is used to split the 
        data according given value(s). If ratio = a (flaot), then test will 
        be a% of total data size.If ratio = [a,b] where a and b are 
        in (0,1), train, test and validation sets are formed according to 
        them. If a = 0.2 and b = 0.1, train fraction is 0.7, test fraction 
        is 0.2 and validation fraction is 0.1 of all dataset size. 
    
    Returns:
        X_train: {numpy array}: training dataset
        X_test: {numpy array}: test dataset
        X_validation: {numpy array}: validation dataset, returns if ratio is list
        y_train: {numpy array}: training dataset's labels
        y_test: {numpy array}: test dataset's labels
        y_validation: {numpy array}: validation dataset's labels, returns if ratio is list
    '''
    if X is not None and y is not None:
        return separator(X,y, ratio)

    elif X is not None and y is None:
        raise ValueError('While X is not None, y cannot be None.')

    elif X is None and X_pos is not None:
        if type(X_pos) == np.ndarray:
            X_pos = X_pos.tolist()
            X_neg = X_neg.tolist()
        y_pos = [1 for i in range(len(X_pos))]
        y_neg = [-1 for i in range(len(X_neg))]

        X = X_pos + X_neg
        y = y_pos + y_neg
        trdn = list(zip(X,y))
        random.shuffle(trdn)
        X,y = zip(*(trdn))

        return separator(X,y, ratio)

    elif X is None and X_neg is None and X_pos is None:
        raise ValueError('X, X_pos and X_neg data cannot be None at the same time.')
    
