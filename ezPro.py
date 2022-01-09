# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 16:30:59 2022

@author: Sameitos
"""

import argparse
import numpy as np
from profab.model_process import *
from profab.model_evaluate import *
from profab.import_dataset import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='ProFAB in terminal to train GO and EC terms')
parser.add_argument('--file_name',
                    type = str,
                    help='File includes dataset names such as GO_0000018, GO_1905523',
                    required=True)
parser.add_argument('--set_type',
                    type = str,
                    default='random',
                    help='Splitting type of train and test sets',
                    )
parser.add_argument('--protein_feature',
                    type=str,
                    default='paac',
                    help='Numerical feature of protein sequence')
parser.add_argument('--ratio',
                    default = 0.2,
                    help = 'Ratio of between validation and test sets.')
parser.add_argument('--ml_type',
                    type = str,
                    default = 'logistic_reg',
                    help = 'Machine learning algorithms will be used in prediction')
parser.add_argument('--score_path',
                    type = str,
                    default='score_path.csv',
                    help = 'A destination where scores are saved. It must be .csv file.')
parser.add_argument('--scale_type',
                    type = str,
                    default = 'standard',
                    help = 'Scaling of data to prevent biases.')
parser.add_argument('--pre_determined',
                    type = bool,
                    default = False,
                    help='If True, data will be given train test splitted else splliting will be done')



def imp_train_result(data_name, **kwargs):
    
    
    data_model = None
    if data_name[:2] == 'GO':
        
        data_model = GOID(ratio = kwargs['ratio'],
                          protein_feature = kwargs['protein_feature'],
                          pre_determined = kwargs['pre_determined'],
                          set_type = kwargs['set_type'])
    
    elif data_name[:2] == 'ec':
        data_model = ECNO(ratio = kwargs['ratio'],
                          protein_feature = kwargs['protein_feature'],
                          pre_determined = kwargs['pre_determined'],
                          set_type = kwargs['set_type'])
    
    if kwargs['set_type'] == 'temporal' or type(
            kwargs['ratio']) == list or type(
                kwargs['ratio']) == float and kwargs[
                    'pre_determined'] == True:
                      
        print(f'Importing data...')
        X_train,X_test,X_validation,y_train,y_test,y_validation = data_model.get_data(
            data_name = data_name)
        
        X_train,scaler = scale_methods(X_train,scale_type = kwargs['scale_type'])
        X_test,X_validation = scaler.transform(X_test),scaler.transform(X_validation)
        
        print(f'Training starts...')
        model = classification_methods(ml_type = kwargs['ml_type'],
                                X_train = X_train,
                                y_train = y_train,
                                X_valid = X_validation,
                                y_valid = y_validation,
                                )
        
        print(f'Predicting test-validation sets labels and Scoring...')
        score_train = evaluate_score(model,X_train,y_train,preds = False)
        score_test = evaluate_score(model,X_test,y_test,preds = False)
        score_validation = evaluate_score(model,X_validation,y_validation,preds = False)
    
        print(f'Training and scoring is done {data_name}\n---------***---------\n')
        return {'train':score_train,'test':score_test,'validation': score_validation}
    
    else:
        
        print(f'Importing data...')
        X_train,X_test,y_train,y_test = data_model.get_data(
            data_name = data_name)
        X_train,scaler = scale_methods(X_train,scale_type = kwargs['scale_type'])
        X_test= scaler.transform(X_test)
        
        print(f'Training starts...')
        model = classification_methods(ml_type = kwargs['ml_type'],
                                X_train = X_train,
                                y_train = y_train,
                                )
        print(f'Predicting test set labels and Scoring...')
        score_train = evaluate_score(model,X_train,y_train,preds = False)
        score_test = evaluate_score(model,X_test,y_test,preds = False)
        
        print(f'Training and scoring is done for {data_name}\n')
        return {'train':score_train,'test':score_test}
        
    

def loop_trough(file_name, **kwargs):
    
    data_names = np.genfromtxt(file_name,dtype = str)
    score_dict = {}
    print('Action is starting...')
    for data_name in data_names:
        print('---------***---------\n')
        print(f'Dataset: {data_name}')
        score_dict.update({data_name:imp_train_result(data_name, **kwargs)})
    
    print(f'Scores are written to score path: {kwargs["score_path"]}\n\n---------***---------\n\n')
    multiple_form_table(score_dict, score_path = kwargs['score_path'])
    
        
        
if __name__ == '__main__':
    args = parser.parse_args()
    loop_trough(args.file_name,
                ml_type = args.ml_type,
                scale_type = args.scale_type,
                score_path = args.score_path,
                ratio = args.ratio,
                protein_feature = args.protein_feature,
                pre_determined = args.pre_determined,
                set_type = args.set_type,
                )
    















