
import os
import sys
import pickle

from bio_loading_features_indices import get_final_dataset
from bio_spliting_means import splitting
from bio_to_table import form_table

sys.path.append('../../scale_learn_evaluate')
from scaler import scaling
from regressions import regression_methods
from classifications import classification_methods 
from evaluation_metrics import evaluate_score

def forming_path(dataset_name, learning_type,machine_type,split_type):
    '''

    Parameters
    ----------
    dataset_name: folder name where data is stored
    learning_type, {regression,classification}, default: regression
    machine_type: machine learning type name
        
    Returns
    -------
    model_path: path where machine learning parameters are stored in bytes 
    score_path: path where scores of machine are stored

    '''
    model_path = dataset_name + '/Model_Data' + '/' + learning_type
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    score_path = dataset_name + '/Score_Data' + '/' + learning_type   
    if not os.path.exists(score_path):
        os.makedirs(score_path)
     
    return model_path + '/' + split_type + '_' +  machine_type + ".txt",score_path + '/' + machine_type + ".csv" 

def model_training_scoring(dataset_name,
         split_type = 'random_split',
         scaler_type = 'minmax',
         learning_type = 'Regression',
         machine_type = 'SVR',
         ratio = 0.2,
         cv = None):
    """
    
    dataset_name: dataset_name name that models and scores are exported.
    
    random_split,{random_split, compound_split, target_split, compound_target_split},
                (default='random_split'): parameter to decide make random splitting or other options
                
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
    return 
    model_path,score_path = forming_path(dataset_name, learning_type,machine_type,split_type)

    dataset = get_final_dataset(dataset_name)
    
    X_train_s,X_test_s,X_validation_s,y_train,y_test,y_validation = splitting(dataset_name,
                                                                              dataset,
                                                                              split_type,
                                                                              ratio)
    X_train,scaler = scaling(X_train_s,scaler_type)
    X_test,X_validation = scaler.transform(X_test_s),scaler.transform(X_validation_s) 


    if not os.path.isfile(model_path):
        if learning_type == 'Regression':regression_methods(model_path,machine_type,X_train,y_train,cv)    
        elif learning_type == 'Classification':classification_methods(model_path,machine_type,X_train,y_train,cv)

    model = pickle.load(open(model_path,'rb'))
    
    S_train,_ = evaluate_score(model,X_train,y_train)
    S_test,_ = evaluate_score(model,X_test,y_test)
    S_validation,_ = evaluate_score(model,X_validation,y_validation)
    
    score = [S_train,S_test,S_validation]
    size = [str(X_train.shape),str(X_test.shape),str(X_validation.shape)]


    form_table(score_path,split_type,score,size)
