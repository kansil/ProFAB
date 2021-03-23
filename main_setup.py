# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:11:55 2021

@author: Sameitos
"""



import os
import pickle

from .utils import separator
from .utils import get_final_dataset
from .utils import form_table

from process_learn_evaluate import scale_methods
from process_learn_evaluate import classification_methods,regression_methods
from process_learn_evaluate import evaluate_score


def forming_path(dataset_name,machine_type,protein_feature):
    '''
    Parameters
    ----------
    dataset_name: folder name where data is stored
    machine_type: machine learning type name
    split_type: split type of dataset
        
    Returns
    -------
    model_path: path where machine learning parameters are stored in bytes 
    score_path: path where scores of machine are stored
    '''
    model_path = dataset_name + '/Model' + '_' + protein_feature + '_' + machine_type + ".txt"

    score_path = dataset_name + '/Score' + '_' + protein_feature + '_' + machine_type + ".csv" 

    return model_path,score_path

def model_all(dataset_name,
          scaler_type = 'minmax',
          learning_type = 'Regression',
          machine_type = 'SVM',
          ratio = 0.2,
          cv = None):
    """
    Description: This function train the datasets through some machine learning algorithms and evaluate
                  extracted model. Then, it forms an table and saves the metrics to csv file.
    Parameters:
                 
        dataset_name: dataset_name name that models and scores are exported.
                    
        scaler_type, {'Normalizer','Standard_Scaler','MaxAbs_Scaler','MinMax_Scaler','Robust_Scaler'}, 
                    (default='minmax'): scaler type name
        
        learning_type, {'Regression, Classification'}, default='Regression': learning type name
        
        machine_type, {'log_reg','ridge_class','KNN','SVM','random_forest','DeepNN','naive_bayes',
                        decision_tree',gradient_boosint'} default: 'random_forest': machine learning type 
                        
        ratio: Train test and validation sets split ratios. If float, train and test sets will be formed,
            If list with size = 2, ratio = [test_ratio,validation_ratio], train, test and validation sets
            will be formed. If datasets are already diveded in train and test, 
            ratio must be None, or float. If float, validation set will be formed via train set. 
            If None, train and test data will be used to train and calculate scores.
        cv, (default: None): cross_validation which can be determined by user. If left None, 
                              RepeatedKFold() function will be applied to use in RandomizedSearch() function
           
    """
    if learning_type != 'Regression' or learning_type != 'Classification':
        raise AttributeError(f'Learning type is not true. Options: Regression, Classification')
    if type(ratio) != list or type(ratio) != float:
        raise AttributeError(f'ratio type is not applicable. Options: float or List')
    
    
    model_path,score_path = forming_path(dataset_name, machine_type, protein_feature)
    
    data = get_final_dataset(dataset_name,learning_type = learning_type)
    
    if len(data) == 2:
        X,y = data
        if type(ratio) == float:
            X_train,X_test,y_train,y_test = separator(X,y, ratio = ratio)
        else:
            X_train,X_test,X_validation,y_train,y_test,y_validation = separator(X,y, ratio = ratio)
    
    elif len(data) == 4:
        #if pre_determined dataset, ratio should be None, Float
        if ratio == None:
            X_train,X_test,y_train,y_test = data
        else:
            X_train,X_test,y_train,y_test = data
            X_train,X_validation,y_train,y_validation = separator(X_train,y_train,ratio = ratio)
         
    X_train,scaler = scale_methods(X_train,scale_type = scaler_type)
    
    if not os.path.isfile(model_path):
        if learning_type == 'Regression':regression_methods(path = model_path,
                                                            ml_type = machine_type,
                                                            X_train = X_train,
                                                            y_train = y_train,
                                                            cv = cv)    
        elif learning_type == 'Classification':classification_methods(path = model_path,
                                                            ml_type = machine_type,
                                                            X_train = X_train,
                                                            y_train = y_train,
                                                            cv = cv)
    
    model = pickle.load(open(model_path,'rb'))
    
    score_train,f_train = evaluate_score(model,X_train,y_train)

    try:
        score_test,f_test = evaluate_score(model,X_test,y_test)
        score_validation,f_validation = evaluate_score(model,X_validation,y_validation)
        scores = [score_train,score_test,score_validation]
        size_of = [str(X_train.shape),str(X_test.shape),str(X_validation.shape)]
        preds = [f_train,f_test,f_validation]
        names = ['Train,Test','Validation']
        form_table(names = names,scores = scores,
                    sizes = size_of, learning_type = learning_type,
                    preds = preds)
    except:
        score_test,f_test = evaluate_score(model,X_test,y_test)
        scores = [score_train,score_test]
        size_of = [str(X_train.shape),str(X_test.shape)]
        preds = [f_train,f_test]
        names = ['Train','Test']
        form_table(score_path = score_path, names = names, scores = scores,
                    sizes = size_of, learning_type = learning_type,
                    preds = preds)

