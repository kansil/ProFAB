# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 00:30:06 2021

@author: Sameitos
"""

import io
import os
import re
import sys
import random
import requests
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from zipfile import ZipFile

import torch

def download_data(server_path,save_path):
    
    '''
    Description:
        Download dataset from Kansil server
    Parameters:
        server_path: {string}, path of server where data is hold
        save_path: {string}, path to save dataset in local
    '''
    
    headers=requests.head(server_path).headers
    #print(headers)
    downloadable = 'Content-Length' in headers.keys()
    if downloadable:
        response = requests.get(server_path, stream = True)
        
        total_byte= int(response.headers.get('content-length', 0)) 
        progress_bar = tqdm(total=total_byte, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(chunk_size= 8*1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    else:
        print("No given dataset is available in server.")
        sys.exit(1)

def separator(X,y,ratio):
    '''
    Description:
        To split data into train, test and validation sets with respect to ratio
        value
    Paramters:
        ratio: {float, list}, used to split data into train, test, validation
            sets as given values. If ratio = a (float), then test will be a%
            of total data size. If ratio = [a,b] where a and b are in (0,1), 
            train, test and validation sets are formed according to them. For
            example, If a = 0.2 and b = 0.1, train fraction is 0.7, test
            fraction is 0.2 and validation fraction is 0.1 of all dataset size.
    Returns:
        X_train: {numpy array}: training dataset
        X_test: {numpy array}: test dataset
        X_validation: {numpy array}: validation dataset, returns if ratio is
            list
        y_train: {numpy array}: training dataset's labels
        y_test: {numpy array}: test dataset's labels
        y_validation: {numpy array}: validation dataset's labels, returns if
            ratio is list
    '''
    if type(ratio) == float:

        return train_test_split(X,y, test_size = ratio)

    elif type(ratio) == list:

        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = ratio[0])

        X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,
                                                                     test_size = ratio[1]/(
                                                                         1-ratio[0]))
            
        return X_train,X_test,X_validation,y_train,y_test,y_validation
    

def _classif_data_import(zip_data,pos_file,neg_file, label, pos_indices = None,neg_indices = None):
    
    '''
    Description:
        Load data from zip file
    Paramters:
        zip_data: {string}, name of zip file
        pos_file: {string}, name of positive data in zip file
        neg_file: {string}, name of negative data in zip file
        label: {'positive','negative'}, if 'negative', only negative set is loaded,
                If 'positive', only positive set is loaded.
        pos_indices: {set}, (default = None), If None, all data is loaded,
            otherwise data at pos_indices are loaded
        neg_indices: {set}, (default = None), If None, all negative data is loaded,
            otherwise negative data at neg_indices are loaded
            
    '''    
    
    pX,nX,X,y = [],[],[],[]
    
    with ZipFile(zip_data) as f:
        pf =  f.open(pos_file)
        nf = f.open(neg_file)
        
        if pos_indices is not None and neg_indices is not None:
	        pos_idx = set([int(i.decode('utf-8').strip('\n')) for i in f.open(pos_indices)])
	        neg_idx = set([int(i.decode('utf-8').strip('\n')) for i in f.open(neg_indices)])
	        for k,(rowx) in enumerate(pf):
	            if k in pos_idx:
	                rowx = re.split('\t',rowx.decode('utf-8').strip('\n'))
	                rowx = list(np.array(rowx[1:],dtype = 'float64'))
	                if label == 'positive':
	                    pX.append(rowx)
	                if label == None:
	                    X.append(rowx)
	                    y.append(1)
	        for k,(rowx) in enumerate(nf):
	            if k in neg_idx:
	                rowx = re.split('\t',rowx.decode('utf-8').strip('\n'))
	                rowx = list(np.array(rowx[1:],dtype = 'float64'))
	                if label == 'negative':
	                    nX.append(rowx)
	                if label == None:
	                    X.append(rowx)
	                    y.append(-1)
        else:

            for rowx in pf:
                rowx = re.split('\t',rowx.decode('utf-8').strip('\n'))
                rowx = list(np.array(rowx[1:],dtype = 'float64'))
                if label == 'positive':
                    pX.append(rowx)
                    
                if label == None:
                    X.append(rowx)
                    y.append(1)
            for rowx in nf:
                rowx = re.split('\t',rowx.decode('utf-8').strip('\n'))
                rowx = list(np.array(rowx[1:],dtype = 'float64'))
                if label == 'negative':
                    nX.append(rowx)
                    
                if label == None:
                    X.append(rowx)
                    y.append(-1)
    pf.close()
    nf.close()
        
    return pX,nX,X,y

def self_data(file_name, delimiter, label, name):
        
    '''
    Description:
        This function is to provide users to import their datasets with
        specified delimiter. The format of data should be like that if 
        delimiter is comma separated and name == True:

            Name(or ID),feature_1,feature_2,...,feature_n
            Name(or ID),feature_1,feature_2,...,feature_n
            Name(or ID),feature_1,feature_2,...,feature_n

    Parameters:
        delimiter: default = "\t", a character to separate columns in file.
        name: type = bool, default = False, If True, then first colmun
            is considered as name of inputs else the first column is a 
            feature column.
        label: type = bool, default = False, If True, then last colmun
            is considered as label of inputs else the last column is a 
            feature column. 
    '''
    try:
        return torch.load(file_name)
    except:
        with open(file_name, 'r') as f:
                
            if label:
                
                X_pos,X_neg = [],[]
                for row in f:
                    row = re.split(delimiter,row.strip())
                    if name:
                        if int(row[-1]) == 1:
    
                            X_pos.append(row[1:-1])
                        else:
                            X_neg.append(row[1:-1])
    
                    else:
                        if int(row[-1]) == 1:
    
                            X_pos.append(row[:-1])
                        else:
                            X_neg.append(row[:-1])
                
                return X_pos,X_neg
            else:
                X = []
                for row in f:
                    row = re.split(delimiter,row.strip())
                    
                    if name:
                            X.append(row[1:])
                    else:
                        X.append(row)
                return X
    

def _classif_form_table(scores, score_path = 'score_path.csv'):
    '''
    Description:
        Storing classification scoring metrics in .csv format
    Paramters:
        scores: {dict}, includes all classification scoring metrics
        score_path: {string}, (default = "score_path.csv"), a path where
            metrics are saved    
    '''
    func = 'w'
    if os.path.isfile(score_path):
        print(f'File {score_path} already exists, Scores are append to old score path')
        func = 'a'
    
    if type(scores) is not dict:
        raise TypeError('Type "scores" should be dictionary')
    
    f = open(score_path,func)
    
    scores.values()
    columns = ['Set'] + list(list(scores.values())[0].keys())
    f.write(f'{",".join(columns)}\n')
    
    for sc in scores.keys():
        score = np.array([sc] + list(scores[sc].values()),dtype = str)          
        f.write(f'{",".join(score)}\n')
    
    f.write(f'\n')
    f.close()
    
def _rgr_form_table(scores, size = None, score_path = 'score_path.csv'):
    '''
    Description:
        Storing regression scoring metrics in .csv format
    Paramters:
        scores: {dict}, includes all regression scoring metrics
        score_path: {string}, (default = "score_path.csv"), a path where
            metrics are saved    
    '''
    func = 'w'
    if os.path.isfile(score_path):
        print(f'File {score_path} already exists, Scores are append to old score path')
        func = 'a'
        

    if type(scores) is not dict:
        raise TypeError('Type "scores" should be dictionary')
    
    f = open(score_path,func)
    columns = ['Set'] + list(list(scores.values())[0].keys())[:-1] + list(
        
        list(scores.values())[0]['threshold based Metrics'].keys())
    
    f.write(f'{",".join(columns)}\n')
    
    for sc in scores.keys():   
                   
        score = np.array([sc] + list(scores[sc].values())[:-1] + list(
            scores[sc]['threshold based Metrics'].values()) ,dtype = str)
        f.write(f'{",".join(score)}\n')

    f.write(f'\n')
    f.close()

def multiform_table(score_dict, score_path):
    '''
    Description:
        Storing classification scoring metrics in .csv format for different
        dataset
    Paramters:
        score_dict: {dict}, includes all classification scoring metrics of
            different datasets
        score_path: {string}, (default = "score_path.csv"), a path where
            metrics are saved    
    '''
    func = 'w'
    if os.path.isfile(score_path):
        print(f'File {score_path} already exists, Scores are append to old score path')
        func = 'a'
        
    f = open(score_path, func)
    
    datasets = list(score_dict.keys())
    columns = ['Dataset Name', 'Set'] + list(list(score_dict[datasets[0]].values())[0].keys())
    f.write(f'{",".join(columns)}\n')
    for data_name in datasets:
        
        f.write(f'{data_name}')
        scores = score_dict[data_name]
        
        for sc in scores.keys():
            
            score = np.array([" "] + [sc] + list(scores[sc].values()),dtype = str)          
            f.write(f'{",".join(score)}\n')
        f.write(f'\n')
    
    f.write(f'\n')
    f.close()
    



#!!!This function is out of use until dti datasets will be active!!!
'''
def _rgr_data_import(zip_data,xf,yf,indices_file = None):
    X,y = [],[]
    with ZipFile(zip_data) as f:
        ready_indices = set([int(i.decode('utf-8').strip('\n')) for i in f.open(indices_file)])
        xff =  f.open(xf)
        yff = f.open(yf)
        if indices_file is not None:
            for k,(rowx,rowy) in enumerate(zip(xff,yff)):
                if k in ready_indices:
                    rowx = re.split(' ',rowx.strip('\n'))
                    rowx = list(np.array(rowx,dtype = 'float64'))
                    X.append(rowx)
                    y.append(rowy.strip('\n'))
    
        else:
            for rowx,rowy in zip(xff,yff):
            
                rowx = re.split(' ',rowx.strip('\n'))
                rowx = list(np.array(rowx,dtype = 'float64'))
                X.append(rowx)
                y.append(rowy.strip('\n'))
    xf.close()
    yf.close()
    
    return X,y


'''















