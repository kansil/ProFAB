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

def download_data(server_path,save_path):
    
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
    
    if type(ratio) == float:

        return train_test_split(X,y, test_size = ratio)

    elif type(ratio) == list:

        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = ratio[0])

        X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,
                                                                     test_size = ratio[1]/(
                                                                         1-ratio[0]))
            
        return X_train,X_test,X_validation,y_train,y_test,y_validation
    
    
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
    

def _classif_data_import(zip_data,pos_file,neg_file, label, pos_indices = None,neg_indices = None):
    
    
    pX,py,nX,ny,X,y = [],[],[],[],[],[]
    
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
	                    py.append(1)
	                if label == None:
	                    X.append(rowx)
	                    y.append(1)
	        for k,(rowx) in enumerate(nf):
	            if k in neg_idx:
	                rowx = re.split('\t',rowx.decode('utf-8').strip('\n'))
	                rowx = list(np.array(rowx[1:],dtype = 'float64'))
	                if label == 'negative':
	                    nX.append(rowx)
	                    ny.append(-1)
	                if label == None:
	                    X.append(rowx)
	                    y.append(-1)
        else:

            for rowx in pf:
                rowx = re.split('\t',rowx.decode('utf-8').strip('\n'))
                rowx = list(np.array(rowx[1:],dtype = 'float64'))
                if label == 'positive':
                    pX.append(rowx)
                    py.append(1)
                if label == None:
                    X.append(rowx)
                    y.append(1)
            for rowx in nf:
                rowx = re.split('\t',rowx.decode('utf-8').strip('\n'))
                rowx = list(np.array(rowx[1:],dtype = 'float64'))
                if label == 'negative':
                    nX.append(rowx)
                    ny.append(-1)
                if label == None:
                    X.append(rowx)
                    y.append(-1)
    pf.close()
    nf.close()
        
    return pX,py,nX,ny,X,y

def self_data(file_name, delimiter, label, name):
        
    data = []
    with open(file_name, 'r') as f:
            
        if label:
            y = []
            for row in f:
                row = re.split(delimiter,row.strip())
                if name:
                    data.append(row[1:-1])
                    y.append(row[-1])

                else:
                    data.append(row[:-1])
                    y.append(row[-1])
            return data,y
        else:
            for row in f:
                row = re.split(delimiter,row.strip())
                
                if name:
                        data.append(row[1:])
                else:
                    data.append(row)
            return data


def _classif_form_table(scores, score_path = 'score_path.csv'):
    
    if type(scores) is not dict:
        raise TypeError('Type "scores" should be dictionary')
    f = open(score_path,'w')
    
    scores.values()
    columns = ['Set'] + list(list(scores.values())[0].keys())
    f.write(f'{",".join(columns)}\n')
    
    for sc in scores.keys():
        score = np.array([sc] + list(scores[sc].values()),dtype = str)          
        f.write(f'{",".join(score)}\n')
    
    f.close()
    
def _rgr_form_table(scores, size = None, score_path = 'score_path.csv'):
    
    if type(scores) is not dict:
        raise TypeError('Type "scores" should be dictionary')
    
    f = open(score_path,'w')
    columns = ['Set'] + list(list(scores.values())[0].keys())[:-1] + list(
        
        list(scores.values())[0]['threshold based Metrics'].keys())
    
    f.write(f'{",".join(columns)}\n')
    
    for sc in scores.keys():   
                   
        score = np.array([sc] + list(scores[sc].values())[:-1] + list(
            scores[sc]['threshold based Metrics'].values()) ,dtype = str)
        f.write(f'{",".join(score)}\n')

    f.close()

def multiform_table(score_dict, score_path):
    
    f = open(score_path, 'w')
    
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
    f.close()
    
    