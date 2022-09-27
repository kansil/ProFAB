# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 23:00:39 2021

@author: Sameitos
"""

from ..utils.imp_split_form import _classif_form_table, _rgr_form_table, multiform_table

def form_table(scores, learning_method = 'classif',path = 'score_path.csv'):

    '''
    Description:
        This function saves performance results to a .csv file.
    Parameters:
        scores: A dictionary that includes set and their scores
        learning_method: {'classif,'rgr}, default = 'classif', Type of
                        learnign method classification or regression
        path: default = 'score_path.csv', A destination where scores are 
                        saved. It must be .csv file.
    '''

    form_methods = {'classif':_classif_form_table,'rgr':_rgr_form_table}
    form_methods[learning_method](scores = scores, score_path = path)    
    
def multiple_form_table(score_dict, score_path = 'score_path.csv'):
    
    '''
    Description:
        This function is automatically forming score table
        by multiple input. No suitable for single input.
    Parameters:
        score_dict: A dict of scores includes results of multiple data,
        score_path: 'score_path.csv', A destination where scores are 
                        saved. It must be .csv file.
    '''
    
    multiform_table(score_dict, score_path)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
