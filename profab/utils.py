# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 00:30:06 2021

@author: Sameitos
"""

import os
import re
import random
import numpy as np
from sklearn.model_selection import train_test_split

class file_to_data():

    def __init__(self, delimiter):
        self.delimiter = delimiter
        

    def _rgr_feature_label(self,file_feature_label):

        f = open(file_feature_label)
        feature,label = [],[]
        for row in f:
            row = re.split(self.delimiter,row.strip('\n'))
            feature.append(list(np.array(row[:-1],dtype = 'float64')))
            label.append(float(row[-1]))

        return feature,label

    def _rgr_feature_label_divided(self,file_feature,file_label):

        feature,label = [],[]
        
        xf = open(file_feature)
        yf = open(file_label)
        
        for row,l in zip(xf,yf):
            row = re.split(self.delimiter,row.strip('\n'))
            feature.append(list(np.array(row,dtype = 'float64')))
            label.append(float(l))

        return feature,label

    def _classif_feature_label(self,file_feature_label):

        f = open(file_feature_label)
        feature,label = [],[]
        for row in f:
            row = re.split(self.delimiter,row.strip('\n'))
            feature.append(list(np.array(row[:-1],dtype = 'float64')))
            label.append(int(row[-1]))

        return feature,label
        

    def _classif_feature_label_divided(self,file_feature,file_label):

        feature,label = [],[]
        
        xf = open(file_feature)
        yf = open(file_label)
        
        for row,l in zip(xf,yf):
            row = re.split(self.delimiter,row.strip('\n'))
            feature.append(list(np.array(row),dtype = 'float64'))
            label.append(int(l))

        return feature,label


    def _classif_pos_neg(self,file_pos,file_neg):

        feature,label = [],[]
        
        pf = open(file_pos)
        nf = open(file_neg)
        
        for row in pf:
            row = re.split(self.delimiter,row.strip('\n'))
            feature.append(list(np.array(row,dtype = 'float64')))
            label.append(1)

        for row in nf:
            row = re.split(self.delimiter,row.strip('\n'))
            feature.append(list(np.array(row,dtype = 'float64')))
            label.append(-1)            

        rdn = list(zip(feature,label))
        random.shuffle(rdn)
        return zip(*rdn)    

def get_final_dataset(dataset_name, learning_method = 'classif'):

    if learning_method == 'rgr':
        
        for format_type,delimiter in zip(['tsv','txt','csv'],['\t',' ',',']):
            
            ftd = file_to_data(delimiter)

            file_feature_label = dataset_name + '/feature_label_dataset.' + format_type
            if os.path.isfile(file_feature_label):
                
                return ftd.reg_feature_label(file_feature_label)


            file_feature = dataset_name + '/feature_dataset.' + format_type
            file_label = dataset_name + '/label_dataset.' + format_type
            if os.path.isfile(file_feature) and os.path.isfile(file_label):

                return ftd.reg_feature_label_divided(file_feature,file_label)                

            file_train_feature_label = dataset_name + '/train_feature_label_dataset.' + format_type
            file_test_feature_label = dataset_name + '/test_feature_label_dataset.' + format_type
            if os.path.isfile(file_train_feature_label) and os.path.isfile(file_test_feature_label):
                train_feature,test_feature,train_label,test_label = [],[],[],[]
                
                train_feature,train_label = ftd.reg_feature_label(file_train_feature_label)
                test_feature,test_label = ftd.reg_feature_label(file_test_feature_label)

                return train_feature,test_feature,train_label,test_label

            file_train_feature = dataset_name + '/train_feature_dataset.' + format_type
            file_train_label = dataset_name + '/train_label_dataset.' + format_type
            file_test_feature = dataset_name + '/test_feature_dataset.' + format_type
            file_test_label = dataset_name + '/test_label_dataset.' + format_type
            
            if os.path.isfile(file_train_feature) and os.path.isfile(
                file_train_label) and os.path.isfile(
                file_test_label) and os.path.isfile(file_test_label):

                train_feature,train_label = ftd.reg_feature_label_divided(file_train_feature,file_train_label)
                test_feature,test_label = ftd.reg_feature_label_divided(file_test_feature, file_test_label)

                return train_feature,test_feature,train_label,test_label

            else:
                raise FileNotFoundError(f'Any of the files "{file_feature_label}" or'/
                    '"{file_feature} and {file_label}" or "{file_train_feature_label} and '/
                    '{file_test_feature_label}" or "{file_train_feature}'/
                    'and {file_train_label} and {file_test_feature} and'/
                    '{file_test_label}"'/
                    'could not be found in the dataset folder.'/
                    'Please gives the rights names to files to import')

    if learning_method == 'classif':
        
        for format_type,delimiter in zip(['tsv','txt','csv'],['\t',' ',',']):
            
            ftd = file_to_data(format_type,delimiter)

            file_feature_label = dataset_name + '/feature_label_dataset.' + format_type
            if os.path.isfile(file_feature_label):
                
                return ftd.cls_feature_label(file_feature_label)


            file_feature = dataset_name + '/feature_dataset.' + format_type
            file_label = dataset_name + '/label_dataset.' + format_type
            if os.path.isfile(file_feature) and os.path.isfile(file_label):

                return ftd.cls_feature_label_divided(file_feature,file_label)                

            file_train_feature_label = dataset_name + '/train_feature_label_dataset.' + format_type
            file_test_feature_label = dataset_name + '/test_feature_label_dataset.' + format_type
            if os.path.isfile(file_train_feature_label) and os.path.isfile(file_test_feature_label):
                train_feature,test_feature,train_label,test_label = [],[],[],[]
                
                train_feature,train_label = ftd.cls_feature_label(file_train_feature_label)
                test_feature,test_label = ftd.cls_feature_label(file_test_feature_label)

                return train_feature,test_feature,train_label,test_label

            file_train_feature = dataset_name + '/train_feature_dataset.' + format_type
            file_train_label = dataset_name + '/train_label_dataset.' + format_type
            file_test_feature = dataset_name + '/test_feature_dataset.' + format_type
            file_test_label = dataset_name + '/test_label_dataset.' + format_type
            
            if os.path.isfile(file_train_feature) and os.path.isfile(
                file_train_label) and os.path.isfile(
                file_test_label) and os.path.isfile(file_test_label):

                train_feature,train_label = ftd.cls_feature_label_divided(file_train_feature,file_train_label)
                test_feature,test_label = ftd.cls_feature_label_divided(file_test_feature, file_test_label)

                return train_feature,test_feature,train_label,test_label

            file_pos = dataset_name + '/positive_dataset.' + format_type
            file_neg = dataset_name + '/negative_dataset.' + format_type

            if os.path.isfile(file_pos) and os.path.isfile(file_neg):

                return ftd.cls_pos_neg(file_pos,file_neg)

            file_train_pos = dataset_name + '/positive_train_dataset.' + format_type
            file_train_neg = dataset_name + '/negative_train_dataset.' + format_type
            file_test_pos = dataset_name + '/positive_test_dataset.' + format_type
            file_test_neg = dataset_name + '/negative_test_dataset.' + format_type

            if os.path.isfile(file_train_pos) and os.path.isfile(
                file_train_neg) and os.path.isfile(
                file_test_pos) and os.path.isfile(file_test_neg):

                train_feature,train_label = ftd.cls_pos_neg(file_train_pos,file_train_neg)
                test_feature,test_label = ftd.cls_pos_neg(file_test_pos,file_test_neg)
            
                return train_feature, test_feature, train_label, test_label

            else:
                raise FileNotFoundError(
                    f'Any of the files "{file_feature_label} or '/
                    '{file_feature}" and {file_label}" or "{file_train_feature_label} and '/
                    '{file_test_feature_label}" or "{file_train_feature} and '/
                    '{file_train_label} and {file_test_feature} and '/
                    '{file_test_label}" or "{file_pos} and {file_neg}" or "{file_train_pos} and '/
                    '{file_train_neg} and {file_test_pos} and {file_test_neg}" '/
                    'could not be found in the dataset folder. '/
                    'Please gives the rights names to files to import')
                

def separator(X,y,ratio):
    
    if type(ratio) == float:

        return train_test_split(X,y, test_size = ratio)

    elif type(ratio) == list:

        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = ratio[0])

        X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,
                                                                     test_size = ratio[1]/(
                                                                         1-ratio[0]))
            
        return X_train,X_test,X_validation,y_train,y_test,y_validation
    
    
def _rgr_data_import(xf,yf,indices_file = None):
    ready_indices = set(np.loadtxt(indices_file))
    X,y = [],[]
    xff =  open(xf)
    yff = open(yf)
    if indices_file != None:
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
    

def _classif_data_import(pos_file,neg_file, label, pos_indices = None,neg_indices = None):
    
    
    pX,py,nX,ny,X,y = [],[],[],[],[],[]

    pf =  open(pos_file)
    nf = open(neg_file)
    if pos_indices != None and neg_indices != None:
        pos_idx = set(np.loadtxt(pos_indices))
        neg_idx = set(np.loadtxt(neg_indices))            
        for k,(rowx) in enumerate(pf):
            if k in pos_idx:
                rowx = re.split('\t',rowx.strip('\n'))
                rowx = list(np.array(rowx[1:],dtype = 'float64'))
                if label == 'positive':
                    pX.append(rowx)
                    py.append(1)
                if label == None:
                    X.append(rowx)
                    y.append(1)
        for k,(rowx) in enumerate(nf):
            if k in neg_idx:
                rowx = re.split('\t',rowx.strip('\n'))
                rowx = list(np.array(rowx[1:],dtype = 'float64'))
                if label == 'negative':
                    nX.append(rowx)
                    ny.append(-1)
                if label == None:
                    X.append(rowx)
                    y.append(-1)
    else:
        
        for rowx in pf:
            rowx = re.split('\t',rowx.strip('\n'))
            rowx = list(np.array(rowx[1:],dtype = 'float64'))
            if label == 'positive':
                pX.append(rowx)
                py.append(1)
            if label == None:
                X.append(rowx)
                y.append(1)
        for rowx in nf:
            rowx = re.split('\t',rowx.strip('\n'))
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

def _classif_form_table(scores, score_path = 'score_path.csv'):
    
    if type(scores) is not dict:
        raise TypeError(f'type "scores" should be dictionary')
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
        raise TypeError(f'type "scores" should be dictionary')
    
    f = open(score_path,'w')
    columns = ['Set'] + list(list(scores.values())[0].keys())[:-1] + list(
        
        list(scores.values())[0]['threshold based Metrics'].keys())
    
    f.write(f'{",".join(columns)}\n')
    
    for sc in scores.keys():   
                   
        score = np.array([sc] + list(scores[sc].values())[:-1] + list(
            scores[sc]['threshold based Metrics'].values()) ,dtype = str)
        f.write(f'{",".join(score)}\n')

    f.close()
