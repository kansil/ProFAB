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
from tqdm import tqdm



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
    print('Uniref Proteins are loading...\n')
    unitref_reps = []
    with open(file_name) as f:
        for i,rows in enumerate(tqdm(f)):
            unitref_reps.append(rows.strip('\n\r'))
    return set(unitref_reps)


#```Uniref50 importing```
def pos_UniRef(sreps,IDs):
    
    filtered_pos = []
    for k,i in enumerate(IDs):
        if i in sreps:
            filtered_pos.append(k)   
    return np.array(filtered_pos)

#```non_Enzyme importing```
def non_import(sreps = None):

    print('non_enzyme data is loading...\n')
    non_ecID = []#enzymes dont have ec number but anno score = 5 or 4
    with open( '../ecNo_propagated_data/no_ecNo.txt') as f:
        for h,rows in enumerate(tqdm(f)):
            # if h >= 98000:
            rows = re.split(' ',rows.strip('\n'))
            if sreps != None:
                if rows[0] in sreps:non_ecID.append(rows) 
            else:non_ecID.append(rows) 
            # if h == 120000: break
    return non_ecID

'''Level 1 Pos & Neg Dataset'''




def level1_exporter(folder,level_1_filename,splitter,uniref_name):
    
    ec_class = []
    files1 = os.listdir(level_1_filename) 
    for i in files1:
        with open(level_1_filename + '/' + i) as f:#get_files + 'ecNo_' + str(i+1) + '.txt'
            classes = np.loadtxt(f,dtype = 'str')
            ec_class.append(classes)
            classes = []

    split_method = {'S':'target_split','R':'random_split'}
    print('Dataset splitting is made in: ', split_method[splitter])

    print('Positive set forming...\n')
    if split_method[splitter] == 'target_split':
        
        print('uniref importing... ')
        sreps = get_reps(uniref_name)
        print('len of sreps: ',len(sreps))
        non_enzymes = non_import(sreps = sreps)
        for k,i in enumerate(tqdm(ec_class)):
        
            if not os.path.exists(folder + '/class_' + str(k+1)+'/' + split_method[splitter]):
                os.makedirs(folder + '/class_' + str(k+1)+'/' + split_method[splitter])

            with open(folder + '/class_' + str(k+1)+'/' + split_method[splitter] + '/train_positive_set.txt',
                      'w') as f:
            
                pos_train_set = pos_UniRef(sreps,i[:,0])#return indices
                with open(folder + '/class_' + str(k+1) +'/' + split_method[splitter] + '/validation_positive_set.txt',
                          'w') as sf: 
                    with open(folder + '/class_' + str(k+1) +'/' + split_method[splitter] + '/validation_positive_set.txt','w') as vf: 
                        testPos_idx = np.random.choice(len(pos_train_set),round(len(pos_train_set)*0.1),replace = False)
                        np.savetxt(sf,i[pos_train_set[testPos_idx]],fmt = '%s')
                        i = i[np.delete(pos_train_set,testPos_idx,axis = 0)]
                        validPos_idx = np.random.choice(len(i),round(len(i)*0.2),replace = False)
                        np.savetxt(vf,i[validPos_idx],fmt = '%s')
                        i = np.delete(i,validPos_idx,axis = 0)
                        ec_class[k] = i
                        np.savetxt(f,i,fmt = '%s') 
        sreps = None
    else:
        non_enzymes = non_import()
        for k,i in enumerate(tqdm(ec_class)):
            
            
            if not os.path.exists(folder + '/class_' + str(k+1)+'/' + split_method[splitter]):
                os.makedirs(folder + '/class_' + str(k+1)+'/' + split_method[splitter])
            with open(folder + '/class_' + str(k+1) +'/' + split_method[splitter] + '/train_positive_set.txt','w') as f:
            
                with open(folder + '/class_' + str(k+1) +'/' + split_method[splitter] + '/test_positive_set.txt','w') as tf:

                    with open(folder + '/class_' + str(k+1) +'/' + split_method[splitter] + '/validation_positive_set.txt','w') as vf:                                            

                        testPos_idx = np.random.choice(len(i),round(len(i)*0.1),replace = False)
                        np.savetxt(tf,i[testPos_idx],fmt = '%s')
                        i = np.delete(i,testPos_idx,axis = 0)
                        validPos_idx = np.random.choice(len(i),round(len(i)*0.2),replace = False)
                        np.savetxt(vf,i[validPos_idx],fmt = '%s')
                        i = np.delete(i,validPos_idx,axis = 0)
                        ec_class[k] = i
                        np.savetxt(f,i,fmt = '%s')

    print('Negative Dataset Forming...\n')
    '''Negative Side'''

    for i in tqdm(range(len(ec_class))):

        with open(folder + '/class_' + str(i+1) +'/' + split_method[splitter] + '/train_negative_set.txt','w') as f:
            size_non,size_enz = round(len(ec_class[i])*0.6),round(len(ec_class[i])*0.4/(len(ec_class)-1))
            
            
            non_IndEnzyme = np.random.choice(len(non_enzymes),size_non,replace = False)
            negTest_idx = np.random.choice(len(non_IndEnzyme),len(testPos_idx),replace = False) 
            
            with open(folder + '/class_' + str(i+1) +'/' + split_method[splitter] + '/test_negative_set.txt','w') as tf:
                
                with open(folder + '/class_' + str(i+1) +'/' + split_method[splitter] + '/validation_negative_set.txt','w') as vf:

                    ind_valid = [non_IndEnzyme[i] for i in negTest_idx]
                    
                    np.savetxt(tf,[non_enzymes[i] for i in ind_valid],fmt = '%s')
                    negTest_idx= set(negTest_idx)
                    ind_train = [i for i in non_IndEnzyme if i not in negTest_idx]
                    np.savetxt(f,[non_enzymes[i] for i in ind_train],fmt='%s')
                    
                    for num,j in enumerate(ec_class):
                        if num !=i:
                            if size_enz > len(ec_class[i-6]):size_enz = len(ec_class[j]) 
                            np.savetxt(f,j[np.random.choice(len(j),size_enz,replace = False)],fmt='%s')


                    neg_train_set = np.loadtxt(folder + '/class_' + str(i+1) +'/' + split_method[splitter] + '/train_negative_set.txt',dtype = str)
                    valid_idx = np.random.choice(len(neg_train_set),len(validPos_idx),replace = False)   
                    np.savetxt(vf,neg_train_set[valid_idx],fmt='%s')
                    neg_train_set = np.delete(neg_train_set,valid_idx,axis = 0)

        with open(folder + '/class_' + str(i+1) +'/' + split_method[splitter] + '/train_negative_set.txt','w') as f:
            np.savetxt(f,neg_train_set,fmt='%s')
    
    






'''Level 2,3,4 Pos & Neg Dataset'''

def level234_exporter(level_1_filename,level_2_filename,folder,splitter,uniref_name,sib_no):
    
    #folder: where data stored
    ec_level1 = []
    files1 = os.listdir(level_1_filename) 
    for i in files1:
        with open(level_1_filename + '/' + i) as f:#get_files + 'ecNo_' + str(i+1) + '.txt'
            classes = np.loadtxt(f,dtype = 'str')
            ec_level1.append(classes)
            classes = []

    split_method = {'S':'target_split','R':'random_split'}
    print('Dataset splitting is made in: ', split_method[splitter])

    non_enzymes = non_import()
    
    ec_class = []
    files = os.listdir(level_2_filename) 
    for i in files:
        with open(level_2_filename+'/' + i) as f:#get_files + 'ecNo_' + str(i+1) + '.txt'
            classes = np.loadtxt(f,dtype = 'str')
            ec_class.append(classes)
            classes = []
                  
    #no uniref filtiration
    for k, (file,i) in enumerate(tqdm(zip(files,ec_class))):
        if (len(i)-len(i)*0.1)*0.2<50:continue

        print('The file and number and len of file: {} , {} , {}'.format(file,k,len(i)))

        if not os.path.exists(folder + '/class_' + file[5:-4] +'/' + split_method[splitter]):
            os.makedirs(folder + '/class_' + file[5:-4] +'/' + split_method[splitter])

        #positive side
        with open(folder + '/class_' + file[5:-4] +'/' + split_method[splitter] + '/train_positive_set.txt', 'w') as f:
            
            with open(folder + '/class_' + file[5:-4] +'/' + split_method[splitter] + '/test_positive_set.txt','w') as tf:

                with open(folder + '/class_' + file[5:-4] +'/' + split_method[splitter] + '/validation_positive_set.txt','w') as vf:

                    testPos_idx = np.random.choice(len(i),round(len(i)*0.1),replace = False)
                    np.savetxt(tf,i[testPos_idx],fmt = '%s')
                    i = np.delete(i,testPos_idx,axis = 0)
                    validPos_idx = np.random.choice(len(i),round(len(i)*0.2),replace = False)
                    np.savetxt(vf,i[validPos_idx],fmt = '%s')
                    i = np.delete(i,validPos_idx,axis = 0)
                    np.savetxt(f,i,fmt = '%s')


        pos_len = len(i)
        
        with open(folder + '/class_' + file[5:-4]+'/' + split_method[splitter] + '/train_negative_set.txt','w') as f:
            
            with open(folder + '/class_' + file[5:-4] +'/' + split_method[splitter] + '/test_negative_set.txt','w') as tf:

                with open(folder + '/class_' + file[5:-4] +'/' + split_method[splitter] + '/validation_negative_set.txt','w') as vf:                
                    
                    if pos_len > 10000:
                        size_sibling, size_enz, size_non = round(0.5*pos_len), round(0.25*pos_len/5), round(0.25*pos_len)
                    elif pos_len < 10000 and pos_len > 1000:
                        size_sibling, size_enz, size_non = round(pos_len), round(pos_len/5), round(pos_len)
                    elif pos_len < 1000:
                        size_sibling, size_enz, size_non = round(3*pos_len), round(3*pos_len/5), round(3*pos_len)

                    non_IndEnzyme = np.random.choice(len(non_enzymes),size_non,replace = False)
                    a = np.random.choice(len(non_IndEnzyme),len(testPos_idx),replace = False)
                    ind_valid = [non_IndEnzyme[i] for i in a]
                    np.savetxt(tf,[non_enzymes[i] for i in ind_valid],fmt = '%s')
                    a= set(a)
                    ind_train = [i for i in non_IndEnzyme if i not in a]
                    

                    np.savetxt(f,[non_enzymes[i] for i in ind_train],fmt='%s')

                    for num,l1 in enumerate(ec_level1):
                        if num != int(file[5]):
                            if len(l1)<size_enz: size_enz = len(l1)
                            np.savetxt(f,l1[np.random.choice(len(l1),size_enz)],fmt='%s')    

                    sibling_counter = 0
                    for sibling in files:
                        if sibling[sib_no] !=file[sib_no]:
                            sibling_counter += 1
                    for sibling,c in zip(files,ec_class):
                        if sibling[sib_no] !=file[sib_no]:
                            c = np.array(c)
                            size_sibling = round(size_sibling/sibling_counter)
                            if size_sibling> len(c): size_sibling =  len(c) 
                            np.savetxt(f,c[np.random.choice(len(c),size_sibling,replace = False)],fmt='%s')
                    c = None
       
                    neg_train_set = np.loadtxt(folder + '/class_' + file[5:-4]+'/' + split_method[splitter] + '/train_negative_set' + '.txt',dtype = str)
                    print('neg len: ',len(neg_train_set))
                    print('valid pos len: ',len(validPos_idx))

                    valid_idx = np.random.choice(len(neg_train_set),len(validPos_idx),replace = False)
                    
                    np.savetxt(vf,neg_train_set[valid_idx],fmt='%s')
                    neg_train_set = np.delete(neg_train_set,valid_idx,axis = 0)

        with open(folder + '/class_' + file[5:-4]+'/' + split_method[splitter] + '/train_negative_set' + '.txt','w') as f:
            np.savetxt(f,neg_train_set,fmt='%s')
           
    

similarity = 50
uniref_name = '../../uniref_protein/uniref' + str(similarity) + '_reps.txt'
splitter = 'S'

splits = ['S','R']
level_1_filename = '../ecNo_propagated_data/level_1'
folder = '../EC_level_1'
if not os.path.exists(folder):
    os.makedirs(folder)
#for s in splits:
#    level1_exporter(folder,level_1_filename,s,uniref_name)

level_2_filename = '../ecNo_propagated_data/level_2'
folder = '../EC_level_2'
splitter = 'R'
if not os.path.exists(folder):
    os.makedirs(folder)

#level234_exporter(level_1_filename,level_2_filename,folder,splitter,uniref_name,sib_no)


level_3_filename = '../ecNo_propagated_data/level_3'
folder = '../EC_level_3'
splitter = 'R'
sib_no = 7
if not os.path.exists(folder):
    os.makedirs(folder)

#level234_exporter(level_1_filename,level_3_filename,folder,splitter,uniref_name,sib_no)


level_4_filename = '../ecNo_propagated_data/level_4'
folder = '../EC_level_4'
splitter = 'R'
sib_no = 9
if not os.path.exists(folder):
    os.makedirs(folder)

level234_exporter(level_1_filename,level_4_filename,folder,splitter,uniref_name,sib_no)


