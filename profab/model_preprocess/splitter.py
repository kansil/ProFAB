# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:26:00 2022

@author: Sameitos
"""

from ..utils import separator

def ttv_split(X,y, ratio = 0.2):
    
    '''
    Description:
        This function splits X and y randomly to train, test and validation
        sets according to ratio value
        
    Parameters:
        X: Feature matrix
        y: Label Matrix
        ratio, type = {float,list}, (default = 0.2), is used to split the 
        data according given value(s). If ratio = a (flaot), then test will 
        be a% of total data size.If ratio = [a,b] where a and b are 
        in (0,1), train, test and validation sets are formed according to 
        them. If a = 0.2 and b = 0.1, train fraction is 0.7, test fraction 
        is 0.2 and validation fraction is 0.1 of all dataset size. 
    
    '''
    
    return separator(X,y, ratio)