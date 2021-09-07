# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 23:00:39 2021

@author: Sameitos
"""

from ..utils import _classif_form_table, _rgr_form_table

def form_table(scores, learning_method = 'classif',path = 'score_path.csv'):


    form_methods = {'classif':_classif_form_table,'rgr':_rgr_form_table}

    form_methods[learning_method](scores = scores, score_path = path)    
    
    