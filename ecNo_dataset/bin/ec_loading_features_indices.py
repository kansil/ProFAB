# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:45:38 2020

@author: Sameitos
"""

import os

import numpy as np
import pandas as pd
import csv
import re
import json
from collections import defaultdict


def import_data(dataset_name,clss,split_type):
    print('import data')
    neg_train = np.loadtxt(dataset_name + 'class_'+clss+ '/' + split_type + '/train_negative_set_PAAC.txt',dtype = '<U11')
    pos_train = np.loadtxt(dataset_name + 'class_'+clss+ '/' + split_type + '/train_positive_set_PAAC.txt',dtype = '<U11')
    pos_valid = np.loadtxt(dataset_name + 'class_'+clss+ '/' + split_type + '/validation_positive_set_PAAC.txt',dtype = '<U11')
    neg_valid = np.loadtxt(dataset_name + 'class_'+clss+ '/' + split_type + '/validation_negative_set_PAAC.txt',dtype = '<U11')
    
    return pos_train,pos_valid,neg_train,neg_valid
    
def concanating(pos_train,pos_valid,neg_train,neg_valid):
    
    print('concanate data')
    print(len(pos_train),len(pos_valid),len(neg_train),len(neg_valid))
    
    y_pos_train = np.ones(len(pos_train)).reshape(len(pos_train),1)
    y_pos_valid = np.ones(len(pos_valid)).reshape(len(pos_valid),1)
    y_neg_train = np.ones(len(neg_train)).reshape(len(neg_train),1)*(-1)
    y_neg_valid = np.ones(len(neg_valid)).reshape(len(neg_valid),1)*(-1)
    a = np.append(pos_train,y_pos_train,axis = 1)
    b = np.append(neg_train,y_neg_train,axis = 1)
    
    print(len(a),len(b))
    train = np.append(a,b,axis = 0)
    np.random.shuffle(train)
    
    c = np.append(pos_valid,y_pos_valid,axis = 1)
    d = np.append(neg_valid,y_neg_valid,axis = 1)
    print(len(c),len(d))
    test = np.append(c,d,axis = 0)
    np.random.shuffle(test)
    
    
    print('train',train)
    valid = train[np.random.randint(0,len(train),int(len(train)*0.2))]
    
    
    train_name,X_train,y_train = train[:,0],np.array(train[:,1:-1],dtype = float),np.array(np.array(train[:,-1],dtype = float),dtype = int)
    test_name,X_test,y_test = test[:,0],np.array(test[:,1:-1],dtype = float),np.array(np.array(test[:,-1],dtype = float),dtype = int)
    valid_name,X_valid,y_valid = valid[:,0],np.array(valid[:,1:-1],dtype = float),np.array(np.array(valid[:,-1],dtype = float),dtype = int)
    
    return X_train,X_valid,X_test,y_train,y_valid,y_test,train_name,test_name,valid_name

def get_final_dataset(dataset_name,clss):

    pos_train,pos_test,neg_train,neg_test = import_data(dataset_name,clss)
    
    X_train,X_valid,X_test,y_train,y_valid,y_test,train_name,test_name,valid_name = concanating(pos_train,
                                                                                                pos_test,
                                                                                                neg_train,
                                                                                                neg_test)
    print(' return the data')
    return X_train,X_valid,X_test,y_train,y_valid,y_test,train_name,test_name,valid_name
