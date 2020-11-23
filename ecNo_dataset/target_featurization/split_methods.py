# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:47:34 2020

@author: Sameitos
"""

import numpy as np
import csv 
import json
import re
import os 
import sys



print('leggo')
def get_reps(file_name):  
    '''

    Description: obtains representative protein in indicated unitref protein clusters

    Parameter
    ---------
    file_name: name of the file
    
    Return
    ------
    unitref_reps: Representatives of each protein_cluster in numpy array form

    '''  
    unitref_reps = []
    with open(file_name) as f:
        for i,rows in enumerate(f):
            unitref_reps.append(rows.strip('\n\r'))
            # if i > 100000: break
    return set(unitref_reps)


#```Uniref50 importing```
def pos_UniRef50(sreps,IDs):
    
    filtered_pos = []
    for k,i in enumerate(IDs):
        if i in sreps:
            filtered_pos.append(k)   
    return np.array(filtered_pos)

#```non_Enzyme importing```
def non_import(splitter,sreps):
    print('in non import',sreps)
    non_ecID = []#enzymes dont have ec number but anno score = 5 or 4
    with open( '../ecNo_propagated_data/no_ecNo.txt') as f:
        for h,rows in enumerate(f):
            # if h >= 98000:
            rows = re.split(' ',rows.strip('\n'))
            if h%50000 == 0: print(len(non_ecID))
            # if h%20000 ==0: print('out if: ',rows,' || ',type(rows))
            if rows[-2] == '5' or rows[-2] == '4':
                # if h%20000 ==0: print('in if: ',rows,' || ',type(rows))
                if splitter == 'S':
                    if rows[0] in sreps:non_ecID.append(rows) 
                else:non_ecID.append(rows) 
                # if h == 120000: break
    return non_ecID

'''Level 1 Pos & Neg Dataset'''
'''Positive Side'''

folder = '../EC_level_1'
if not os.path.exists(folder):
    os.makedirs(folder)

def level1_exporter(folder,ec_class,splitter,uniref50_name):
    
    
    print('finish uniref')
    
    
    print('finish import fs')
    split_method = {'S':'similarity_based','R':'random'}
    
    print('reach pos side')
    if split_method[splitter] == 'similarity_based':
        print('in sim')
        
        sreps = get_reps(uniref50_name)
        non_enzymes = non_import(splitter,sreps)
        for k,i in enumerate(ec_class):
        
            if not os.path.exists(folder + '/class_' + str(k+1)+'/' + split_method[splitter]):
                os.makedirs(folder + '/class_' + str(k+1)+'/' + split_method[splitter])
            with open(folder + '/class_' + str(k+1)+'/' + split_method[splitter] + '/train_positive_set' + '.txt',
                      'w') as f:
            
                pos_train_set = pos_UniRef50(sreps,i[:,0])#return indices
                with open(folder + '/class_' + str(k+1) +'/' + split_method[splitter] + '/validation_positive_set' + '.txt',
                          'w') as sf: 
                    a = np.random.choice(len(pos_train_set),round(len(pos_train_set)*0.1))
                    print(a)
                    np.savetxt(sf,i[pos_train_set[a]],fmt = '%s')
                    
                    b = i[np.delete(pos_train_set,a)]
                    ec_class[k] = b
                    np.savetxt(f,b,fmt = '%s') 
        sreps = None
    else:
        print('in random')
        non_enzymes = non_import(splitter,sreps = None)
        for k,i in enumerate(ec_class):
            
            
            if not os.path.exists(folder + '/class_' + str(k+1)+'/' + split_method[splitter]):
                os.makedirs(folder + '/class_' + str(k+1)+'/' + split_method[splitter])
            with open(folder + '/class_' + str(k+1) +'/' + split_method[splitter] + '/train_positive_set' + '.txt',
                      'w') as f:
            
                with open(folder + '/class_' + str(k+1) +'/' + split_method[splitter] + '/validation_positive_set' + '.txt',
                          'w') as sf: 
                    a = np.random.choice(len(i),round(len(i)*0.1))                
                    
                    np.savetxt(sf,i[a],fmt = '%s')
                    
                    b = np.delete(i,a)
                    ec_class[k] = b
                    np.savetxt(f,b,fmt = '%s')
                    
    print('finished pos side')
    print('reach negative side')
    '''Negative Side'''

    for i in range(len(ec_class)):
        
        print('im here' + str(i+6))
        
        with open(folder + '/class_' + str(i+1) +'/' + split_method[splitter] + '/train_negative_set' + '.txt','w') as f:
            size_non,size_enz = round(len(ec_class[i])*0.6),round(len(ec_class[i])*0.4/(len(ec_class)-1))
            
            print(len(non_enzymes))
            
            non_IndEnzyme = np.random.randint(0,len(non_enzymes),size_non)
            a = np.random.choice(len(non_IndEnzyme),round(len(non_IndEnzyme)*0.1)) 
            
            with open(folder + '/class_' + str(i+1) +'/' + split_method[splitter] + '/validation_negative_set' + str(i+1) + '.txt','w') as sf:
                ind_valid = [non_IndEnzyme[i] for i in a]
                
                np.savetxt(sf,[non_enzymes[i] for i in ind_valid],fmt = '%s')
                a= set(a)
                print(sys.getsizeof(a))
                ind_train = [i for i in non_IndEnzyme if i not in a]
                np.savetxt(f,[non_enzymes[i] for i in ind_train],fmt='%s')
                
                np.savetxt(f,ec_class[i-6][np.random.randint(0,len(ec_class[i-6]),size_enz)],fmt='%s')
                np.savetxt(f,ec_class[i-5][np.random.randint(0,len(ec_class[i-5]),size_enz)],fmt='%s')
                np.savetxt(f,ec_class[i-4][np.random.randint(0,len(ec_class[i-4]),size_enz)],fmt='%s')
                np.savetxt(f,ec_class[i-3][np.random.randint(0,len(ec_class[i-3]),size_enz)],fmt='%s')
                np.savetxt(f,ec_class[i-2][np.random.randint(0,len(ec_class[i-2]),size_enz)],fmt='%s')
                np.savetxt(f,ec_class[i-1][np.random.randint(0,len(ec_class[i-1]),size_enz)],fmt='%s')
    print('finished negative side')
    
    
uniref50_name = 'uniref50/uniref50_reps.txt'
get_files = '../ecNo_propagated_data'
splitter = 'R'

ec_class = []
a = 0
files = os.listdir(get_files+ '/level_1') 
for i in files:
    with open(get_files + '/level_1/' + i) as f:#get_files + 'ecNo_' + str(i+1) + '.txt'
        classes = np.loadtxt(f,dtype = 'str')
        print(len(classes))
        ec_class.append(classes)
        classes = []
print('1 finish')

level1_exporter(folder,ec_class,splitter,uniref50_name)
          









