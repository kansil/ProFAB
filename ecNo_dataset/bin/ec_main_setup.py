import os
import numpy as np
import pickle
os.chdir(os.getcwd())

from loading_features_indices import get_final_dataset
# from spliting_means import splitting
from scaler import scaling
from regressions import regression_methods
from classifications import classification_methods 
from evaluation_metrics import evaluate_score
from to_table import form_table


def forming_path(dataset_name, clss, learning_type,machine_type,split_type):
    '''

    Parameters
    ----------
    dataset_name: folder name where data is stored
    learning_type, {regression,classification}, default: regression
    machine_type: machine learning type name
    split_type: split type of 
        
    Returns
    -------
    model_path: path where machine learning parameters are stored in bytes 
    score_path: path where scores of machine are stored

    '''
    model_path = dataset_name + '/class_' + clss + '/' + split_type + '/Model_Data' + '/' + learning_type
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    score_path = dataset_name + '/class_'+ clss + '/' + split_type + '/Score_Data' + '/' + learning_type  
    if not os.path.exists(score_path):
        os.makedirs(score_path)
     
    return model_path + '/' +  machine_type + ".txt",score_path  + '/' + machine_type + ".csv" 

def main(dataset_name,
         clss,
         split_type = 'similarity',
         scaler_type = 'MinMax_Scaler',
         learning_type = 'Classification',
         machine_type = 'SVM',
         ratio = 0.2,
         cv = None):
    """
    
    dataset_name: dataset_name name that models and scores are exported. For example
    
    clss: (str), corresponding enzyme class. For main classes, clss = {1,2,3,4,5,6,7}. For subclasses,
                clss = {1.1,1.2,...,5.1.2.4,...,7.2,...}
    
    split_type, {similarity_based,random},
                (default='similarity_based'): parameter to decide make random splitting or other options
                
    scaler_type, {'Normalizer','Standard_Scaler','MaxAbs_Scaler','MinMax_Scaler','Robust_Scaler'}, 
                (default='minmax'): scaler type name
    
    learning_type, {'Regression, Classification'}, default='Regression': learning type name
    
    machine_type, {'log_reg','ridge_class','KNN','SVM','random_forest','DeepNN','naive_bayes',decision_tree',gradient_boosint'}
              default: 'random_forest': machine learning type name
    ratio: Train test and validation sets split ratios. For ratio:0.2, 
           60% is train %20 test and %20 validation set 
    cv, (default: None): cross_validation which can be determined by user. If left None, 
                         RepeatedKFold() function will be applied to use in RandomizedSearch() function
           
    """
    
    model_path,score_path = forming_path(dataset_name, clss, learning_type,machine_type,split_type)

    X_train_s,X_validation_s,X_test_s,y_train,y_validation,y_test,train_name,test_name,valid_name = get_final_dataset(dataset_name,clss,split_type)

    print('split and improting finished')
    X_train,scaler = scaling(X_train_s,scaler_type)
    X_test,X_validation = scaler.transform(X_test_s),scaler.transform(X_validation_s) 


    print('scaling just finished')
    if not os.path.isfile(model_path):
        if learning_type == 'Regression':regression_methods(model_path,machine_type,X_train,y_train,cv)    
        elif learning_type == 'Classification':classification_methods(model_path,machine_type,X_train,y_train,cv)

    print('modeled already')
    model = pickle.load(open(model_path,'rb'))
    
    
    print('scoring')
    S_train = evaluate_score(model,X_train,y_train)
    S_test = evaluate_score(model,X_test,y_test)
    S_validation = evaluate_score(model,X_validation,y_validation)
    print('scoring finished')
    # if learning_type == 'Classification': 
    scores = [S_train,S_test,S_validation]
    sizes = [str(X_train.shape),str(X_test.shape),str(X_validation.shape)]
    
    print('tabling')
    form_table(score_path,scores,sizes)
    print('done')


dataset_name = '../EC_level_1'
scaler_type = 'MinMax_Scaler',
split_type = 'similarity_based'
learning_type = 'Classification'
machine_type = 'SVM'
clsses = ['1','2','3','4','5','6','7']

for i in clsses:
    
    if __name__ == '__main__':
    
        main(dataset_name,clss = i)


