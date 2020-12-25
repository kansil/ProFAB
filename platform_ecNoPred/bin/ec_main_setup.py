import os
import sys
import numpy as np
import pickle
from ec_loading_features_indices import get_final_dataset
from ec_to_table import form_table

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
    split_type: split type of 
        
    Returns
    -------
    model_path: path where machine learning parameters are stored in bytes 
    score_path: path where scores of machine are stored

    '''
    model_path = dataset_name+ '/Model_Data'+ '/' + split_type 
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    score_path = dataset_name + '/Score_Data' +'/' + split_type 
    if not os.path.exists(score_path):
        os.makedirs(score_path)
     
    return model_path + '/' +  machine_type + ".txt",score_path  + '/' + machine_type + ".csv" 

def main(dataset_name,
         split_type = 'similarity_based',
         scaler_type = 'MinMax_Scaler',
         learning_type = 'Classification',
         machine_type = 'SVM',
         ratio = 0.2,
         cv = None):
    """
    
    dataset_name: dataset_name name that models and scores are exported. For example
    
    split_type, {similarity_based,random},
                (default='similarity_based'): parameter to decide make random splitting or other options
                
    scaler_type, {'Normalizer','Standard_Scaler','MaxAbs_Scaler','MinMax_Scaler','Robust_Scaler'}, 
                (default='minmax'): scaler type name
    
    learning_type, {'Regression, Classification'}, default='Regression': learning type name
    
    machine_type, {'KNN','SVM','random_forest','DeepNN','naive_bayes',decision_tree',gradient_boosint'}
              default: 'random_forest': machine learning type name
    ratio: Train and validation sets split ratios. For ratio:0.2, 
           72% is train %10 test(fixed) and %18 validation set 
    cv, (default: None): cross_validation which can be determined by user. If left None, 
                         RepeatedKFold() function will be applied to use in RandomizedSearch() function
           
    """
    
    model_path,score_path = forming_path(dataset_name,learning_type,machine_type,split_type)

    X_train_s,X_validation_s,X_test_s,y_train,y_validation,y_test,train_name,test_name,valid_name = get_final_dataset(dataset_name,split_type,ratio)

    print('split and improting finished')
    X_train,scaler = scaling(X_train_s,scaler_type)
    X_test,X_validation = scaler.transform(X_test_s),scaler.transform(X_validation_s) 

    print('scaling just finished')
    print('Machine type: ',machine_type)
    print('dataset: ',dataset_name)
    print('model_path: ',model_path)
    if not os.path.isfile(model_path):
        if learning_type == 'Regression':regression_methods(model_path,machine_type,X_train,y_train,cv)    
        elif learning_type == 'Classification':classification_methods(model_path,machine_type,X_train,y_train,cv)

    print('modeled already')
    model = pickle.load(open(model_path,'rb'))
    
    
    print('scoring')
    S_train,f_train = evaluate_score(learning_type,model,X_train,y_train)
    S_test,f_test = evaluate_score(learning_type,model,X_test,y_test)
    S_validation,f_validation = evaluate_score(learning_type,model,X_validation,y_validation)
    print('scoring finished')
 
    scores = [list(S_train.values())[:-4],list(S_test.values())[:-4],list(S_validation.values())[:-4]]
    sizes = [str(X_train.shape),str(X_test.shape),str(X_validation.shape)]
    
    y_preds = [f_train,f_test,f_validation]
    
    with open(score_path[:-4] + '_true_pred.csv', 'w') as f:
        np.savetxt(f,np.array([['Chembl_ID','Predicted labels','True labels']]),fmt = '%s') 
        yss = np.append(test_name.reshape(len(test_name),1),y_test.reshape(len(test_name),1),axis = 1)
        np.savetxt(f,np.append(yss,y_preds[1].reshape(len(y_preds[1]),1),axis = 1),fmt = '%s')

    with open(score_path[:-4] + '_confusion.csv', 'w') as f:
        np.savetxt(f,np.array([['Set','TP','FP','TN','FN']]),fmt = '%s') 
        np.savetxt(f,np.array([['Train'] + list(S_train.values())[-4:]]),fmt = '%s')
        np.savetxt(f,np.array([['Test'] + list(S_test.values())[-4:]]),fmt = '%s')
        np.savetxt(f,np.array([['Validation'] + list(S_validation.values())[-4:]]),fmt = '%s')

    print('len of scores',len(list(S_train.values())[:-4]))
    print('table')
    form_table(score_path,scores,sizes)
    print('done')


dataset_name = '../EC_level_1/class_1'
scaler_type = 'MinMax_Scaler',
split_type = 'similarity_based'
learning_type = 'Classification'
machine_type = 'SVM'
ratio = 0.2
main(dataset_name)
#split_types = ['similarity_based','random']
#machines = ['SVM','naive_bayes','random_forest']
#files = ['../EC_level_1/class_1','../EC_level_1/class_2','../EC_level_1/class_3','../EC_level_1/class_4','../EC_level_1/class_5','../EC_level_1/class_6','../EC_level_1/class_7']
#for i in files:
#    for s in split_types:
#        for m in machines:
#main(dataset_name= i,split_type = s,machine_type = m)

