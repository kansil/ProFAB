# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:45:38 2020

@author: Sameitos
"""

import numpy as np

def import_data(dataset_name,split_type):

    neg_train = np.loadtxt(dataset_name +  '/' + split_type + '/train_negative_feature.txt',dtype = '<U11')
    pos_train = np.loadtxt(dataset_name +  '/' + split_type + '/train_positive_feature.txt',dtype = '<U11')
    pos_test = np.loadtxt(dataset_name +  '/' + split_type + '/test_positive_feature.txt',dtype = '<U11')
    neg_test = np.loadtxt(dataset_name +  '/' + split_type + '/test_negative_feature.txt',dtype = '<U11')
    pos_valid = np.loadtxt(dataset_name +  '/' + split_type + '/validation_positive_feature.txt',dtype = '<U11')
    neg_valid = np.loadtxt(dataset_name +  '/' + split_type + '/validation_negative_feature.txt',dtype = '<U11')

    return pos_train,pos_test,pos_valid,neg_train,neg_test,neg_valid
    
def concanating(pos_train,pos_test,pos_valid,neg_train,neg_test,neg_valid):
    
    y_pos_train = np.ones(len(pos_train)).reshape(len(pos_train),1)
    y_pos_test = np.ones(len(pos_test)).reshape(len(pos_test),1)
    y_pos_valid = np.ones(len(pos_valid)).reshape(len(pos_valid),1)
    y_neg_train = np.ones(len(neg_train)).reshape(len(neg_train),1)*(-1)
    y_neg_test = np.ones(len(neg_test)).reshape(len(neg_test),1)*(-1)
    y_neg_valid = np.ones(len(neg_valid)).reshape(len(neg_valid),1)*(-1)
    
    a = np.append(pos_train,y_pos_train,axis = 1)
    b = np.append(neg_train,y_neg_train,axis = 1)
    
    train = np.append(a,b,axis = 0)
    np.random.shuffle(train)
    
    p = np.append(pos_test,y_pos_test,axis = 1)
    r = np.append(neg_test,y_neg_test,axis = 1)
    
    test = np.append(p,r,axis = 0)
    np.random.shuffle(test)

    c = np.append(pos_valid,y_pos_valid,axis = 1)
    d = np.append(neg_valid,y_neg_valid,axis = 1)

    valid = np.append(c,d,axis = 0)
    np.random.shuffle(valid)
    
    
    train_name,X_train,y_train = train[:,0],np.array(train[:,1:-1],dtype = float),np.array(np.array(train[:,-1],dtype = float),dtype = int)
    test_name,X_test,y_test = test[:,0],np.array(test[:,1:-1],dtype = float),np.array(np.array(test[:,-1],dtype = float),dtype = int)
    valid_name,X_valid,y_valid = valid[:,0],np.array(valid[:,1:-1],dtype = float),np.array(np.array(valid[:,-1],dtype = float),dtype = int)
    
    
    
    #''' randomly assigned validation set'''
    #valid = train[np.random.randint(0,len(train),int(len(train)*0.2))]

    return X_train,X_valid,X_test,y_train,y_valid,y_test,train_name,test_name,valid_name

def get_final_dataset(dataset_name,split_type):

    pos_train,pos_test,pos_valid,neg_train,neg_test,neg_valid = import_data(dataset_name,split_type)
    
    X_train,X_valid,X_test,y_train,y_valid,y_test,train_name,test_name,valid_name = concanating(pos_train,
                                                                                                pos_test,
                                                                                                pos_valid,
                                                                                                neg_train,
                                                                                                neg_test,
                                                                                                neg_valid)
    
    return X_train,X_valid,X_test,y_train,y_valid,y_test,train_name,test_name,valid_name



