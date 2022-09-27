# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 16:30:59 2022

@author: Sameitos
"""

import os, re
import argparse
import numpy as np

from profab.model_learn.classifications import classification_methods
from profab.model_learn.regressions import regression_methods
from profab.model_evaluate.evaluation_metrics import evaluate_score
from profab.model_evaluate.form_table import *
from profab.import_dataset.data_loader import *
from profab.model_preprocess.scaler import scale_methods
from profab.model_preprocess.extracter import extract_protein_feature
from profab.model_preprocess.splitter import ttv_split
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='ProFAB in terminal to train GO and EC terms')
parser.add_argument('--file_name',
                    type = str,
                    help='File includes dataset names such as GO_0000018,'
                    ' GO_1905523. Each name must be defined in new line.',
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
                    type = lambda s: [float(item) for item in s.split(',')],
                    default = [0.2],
                    help = 'Ratio of between validation and test sets.')
parser.add_argument('--ml_type',
                    type = str,
                    default = 'logistic_reg',
                    help = 'Machine learning algorithms will be used in prediction')
parser.add_argument('--score_path',
                    type = str,
                    default='score_path.csv',
                    help = 'A destination where scores are saved. It must be .csv file.')
parser.add_argument('--model_path',
                    type = str,
                    default=None,
                    help = 'A destination where model parameters are saved.')
parser.add_argument('--scale_type',
                    type = str,
                    default = 'standard',
                    help = 'Scaling of data to prevent biases.')
parser.add_argument('--pre_determined',
                    type = bool,
                    default = False,
                    help='If True, data will be given train test splitted else splliting will'
                    ' be done')

parser.add_argument('--isFasta',
                    type = bool,
                    default = False,
                    help='If True, a data provided by user is Fasta file else numerical data'
                    '  should be introduced')
parser.add_argument('--place_protein_id',
                    type = int,
                    default = 0,
                    help = "It indicates the place of protein id in fasta header."
                           " e.g. fasta header: >sp|O27002|....|....|...., seperate the header wrt."
                           " '|' then >sp is in the zeroth position, protein id in the first(1) "
                           "position.")
parser.add_argument('--output_fasta',
                    type = str,
                    default = '',
                    help='Name of folder where output will be saved.')
parser.add_argument('--take_avg',
                    type = bool,
                    default = False,
                    help = 'If False, output will be saved as torch.tensor. If True, average of'
                            'vectors will be saved as array. (arg for NLP methods)')
parser.add_argument('--max_len',
                    type = int,
                    default = -1,
                    help = 'Max sequence lenght to embed (arg for NLP methods)')

parser.add_argument('--isUser',
                    type = bool,
                    default = False,
                    help='If True, user data path must be defined in file else ProFAB data'
                    '  will be used if data names are introduced correctly.')
parser.add_argument('--label',
                    type = bool,
                    default = False,
                    help = "If True, then last colmun"
                            " is considered as label of inputs else the last column is a"
                            " feature column.")
parser.add_argument('--name',
                    type = bool,
                    default = False,
                    help = "If True, then first colmun"
                           " is considered as name of inputs else the first column is a"
                           " feature column.")
parser.add_argument('--delimiter',
                    type = str,
                    default = "\t",
                    help = "A character to separate columns in file.")

def imp_train_result(data_name, model_path, kwargs, user_kwargs, fasta_kwargs):
    
    dataset = ()
    data_model = None
    
        
    if kwargs['isFasta']:
        
        for fasta in os.listdir(data_name):

            output_file = extract_protein_feature(
                protein_feature = kwargs['protein_feature'],
                place_protein_id = fasta_kwargs['place_protein_id'],
                take_avg = True,
                max_len = fasta_kwargs['max_len'],
                
                input_folder = data_name,
                output_folder=kwargs['output_fasta'],
                fasta_file_name = fasta[:-6],
                )
            if re.search('positive',output_file):    
                X_pos_file_name = output_file
            else:
                X_neg_file_name = output_file
        
        
        pPath = os.path.split(os.path.realpath(__file__))[0]
        X_pos = SelfGet(name = True).get_data(
                            pPath + '/' + X_pos_file_name)
        X_neg = SelfGet(name = True).get_data(
                            pPath + '/' + X_neg_file_name)
                            
        datasets = ttv_split(X_pos = X_pos,X_neg = X_neg,ratio = kwargs['ratio'])
        
    elif kwargs['isUser']:
        
        if user_kwargs['label']:
            X_pos,X_neg = SelfGet(label = user_kwargs['label'],
                                  name = user_kwargs['name'],
                                  delimiter = user_kwargs['delimiter']).get_data(
                                      data_name + '/' + os.listdir(data_name))
            
        for dataset in os.listdir(data_name):
            
            if re.search('positive',dataset):
                X_pos = SelfGet(name = user_kwargs['name'],
                                delimiter = user_kwargs['delimiter']).get_data(
                                    data_name + '/' + dataset)
            else:    
                X_neg = SelfGet(name = user_kwargs['name'],
                                delimiter = user_kwargs['delimiter']).get_data(
                                    data_name + '/' + dataset)
        
        datasets = ttv_split(X_pos = X_pos,X_neg = X_neg,ratio = kwargs['ratio'])
        
    else:
            
        
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
            datasets = data_model.get_data(
                data_name = data_name)
        
        else:
            
            print(f'Importing data...')
            datasets = data_model.get_data(
                data_name = data_name)
        
    if len(datasets) == 6:
        X_train,X_test,X_validation,y_train,y_test,y_validation = datasets
        
        X_train,scaler = scale_methods(X_train,scale_type = kwargs['scale_type'])
        X_test,X_validation = scaler.transform(X_test),scaler.transform(X_validation)
        
        print(f'Training starts...')
        model = classification_methods(ml_type = kwargs['ml_type'],
                                X_train = X_train,
                                y_train = y_train,
                                X_valid = X_validation,
                                y_valid = y_validation,
                                path = model_path
                                )
        
        print(f'Predicting test-validation sets labels and Scoring...')
        score_train = evaluate_score(model,X_train,y_train,preds = False)
        score_test = evaluate_score(model,X_test,y_test,preds = False)
        score_validation = evaluate_score(model,X_validation,y_validation,preds = False)
    
        idn = re.split('/',data_name)[-1]
        print(f'Training and scoring is done {idn}\n---------***---------\n')
        return {'train':score_train,'test':score_test,'validation': score_validation}
    
    if len(datasets) == 4:
        
        X_train,X_test,y_train,y_test = datasets
        
        X_train,scaler = scale_methods(X_train,scale_type = kwargs['scale_type'])
        X_test= scaler.transform(X_test)
        
        print(f'Training starts...')
        model = classification_methods(ml_type = kwargs['ml_type'],
                                X_train = X_train,
                                y_train = y_train,
                                path = model_path
                                )
        print(f'Predicting test set labels and Scoring...')
        score_train = evaluate_score(model,X_train,y_train,preds = False)
        score_test = evaluate_score(model,X_test,y_test,preds = False)
        
        idn = re.split('/',data_name)[-1]
        print(f'Training and scoring is done for {idn}\n---------***---------\n')
        return {'train':score_train,'test':score_test}
        
    

def loop_trough(file_name, kwargs, user_kwargs, fasta_kwargs):
    model_path = kwargs['model_path']
    data_names = []
    with open(file_name) as f:
        for row in f:
            if row.strip('\n') != '':
                data_names.append(row.strip('\n'))
    score_dict = {}
    if data_names:
        for data_name in data_names:
            if kwargs['model_path'] is not None:
                model_path = data_name + '_' + kwargs['model_path']
           
            print('---------***---------\n')
            idn = re.split('/',data_name)[-1]
            print(f'Dataset: {idn}')
            score_dict.update({re.split('/',data_name)[-1]:imp_train_result(data_name,
                                                          model_path,
                                                          kwargs,
                                                          user_kwargs,
                                                          fasta_kwargs)})
        
        if score_dict.keys():
            print(f'Scores are written to score path: {kwargs["score_path"]}\n\n'
                  f'---------***---------\n\n')
            multiple_form_table(score_dict, score_path = kwargs['score_path'])
        else:
            print("Evaluation for given datasets could not be done.")
    else:
        raise FileNotFoundError(f'No file name is provided in sample file.')
        
        
if __name__ == '__main__':
    
    args = parser.parse_args()
    
    if args.isUser == True and args.isFasta == True:
        raise AttributeError('Both "isUser" and "isFasta" cannot be True at the same time.')
    
    r = args.ratio
    if len(r) == 1: r = r[0]
    
    
    fasta_kwargs = dict(place_protein_id = args.place_protein_id,
                        max_len = args.max_len)
    
    user_kwargs = dict(delimiter = args.delimiter,
                       name = args.name,
                       label = args.label)
    
    kwargs = dict(
        isUser = args.isUser,
        isFasta = args.isFasta,
        ml_type = args.ml_type,
        scale_type = args.scale_type,
        score_path = args.score_path,
        model_path = args.model_path,
        ratio = r,
        output_fasta = args.output_fasta,
        protein_feature = args.protein_feature,
        pre_determined = args.pre_determined,
        set_type = args.set_type
        ) 
        
    loop_trough(args.file_name,
                kwargs,
                user_kwargs,
                fasta_kwargs)
                
    















