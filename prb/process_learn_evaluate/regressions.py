
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:39:12 2020

@author: Sameitos
"""


import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV,RepeatedKFold,RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
import os
import pickle
from scipy.sparse import csr_matrix
    
class regressors(object):
    """
    Description:In class,6 different machine learning methods for regression are introduced. Their hyperparameters are tuned by
                RandomizedSearchCV and all methods return only their hyperparameters that give the best accoring to cv
                that is created by RepeatedStraitKFold.
      
    """

    def __init__(self,path,cv):
        """
        Parameters: 
	        path: where outcome of training is saved
	        cv: repeated K-Fold Cross Validation 
 
        """
        self.path = path
        self.parameters = None
        self.n_jobs = -1
        self.cv = cv
        
    def get_best_model(self, X_train,y_train,model):
        """
        Parameters
        ----------
        X_train : Feature matrix
        y_train : Label matrix
        model : model specified in ML algorithm functions such as logistic_regressio, SVM, ...
        path : path that model parameters will be saved

        """
        if self.cv == None:
        	
            self.cv = RepeatedKFold(n_splits=10,n_repeats = 5,random_state= 2)
            
            clf = RandomizedSearchCV(model,self.parameters,cv= self.cv, n_jobs=self.n_jobs,scoring="neg_mean_squared_error")
        
        else: 
            clf = RandomizedSearchCV(model,self.parameters,
                                     n_jobs=self.n_jobs,
                                     scoring="neg_mean_squared_error")

        clf.fit(X_train,y_train)
        best_model = clf.best_estimator_
        with open(self.path, 'wb') as f:
            pickle.dump(best_model,f)

    def SVR(self,X_train,y_train):
        from sklearn.svm import SVR
        
        kernel = ["poly","rbf","sigmoid",'sigmoid']
        C = np.linspace(0.1,50,100)
        #nu = np.linspace(0.23,0.4,100)
        gamma = [10**x for x in np.arange(-5,5,dtype = float)]
        max_iter = np.arange(50,200)
        self.parameters = dict(C = C,kernel = kernel,gamma = gamma,max_iter = max_iter)
        model = SVR()
        self.get_best_model(X_train,y_train,model)
        
    def random_forest(self,X_train,y_train):
        from sklearn.ensemble import RandomForestRegressor
        
        n_estimators = [int(i) for i in np.linspace(10,50,num=40)]
        max_features = ["auto","sqrt","log2"]
        bootstrap = [True, False]
        max_depth = np.arange(1,32)
        min_samples_split = np.linspace(0.1, 1.0, 9, endpoint=True)
        self.parameters = dict(n_estimators = n_estimators,max_features = max_features,
                          max_depth=max_depth,min_samples_split = min_samples_split,
                          bootstrap = bootstrap)
        model = RandomForestRegressor()   
        self.get_best_model(X_train,y_train,model)

    def DNN(self,X_train,y_train):
        
        from sklearn.neural_network import MLPRegressor
        
        hidden_layer_sizes = ([int(x) for x in np.arange(100,300)],[int(x) for x in np.arange(1,5)])
        activation = ['logistic', 'tanh', 'relu']
        solver = ['sgd','adam']
        alpha = np.linspace(0.0001,0.001,num = 10)
        batch_size = [100,150]
        learning_rate = ['constant', 'invscaling', 'adaptive']
        max_iter = np.arange(20,50)
        self.parameters = dict(hidden_layer_sizes = hidden_layer_sizes,activation = activation,
                          solver = solver,alpha = alpha,batch_size = batch_size,
                          learning_rate = learning_rate, max_iter = max_iter)
        model = MLPRegressor()
        self.get_best_model(X_train,y_train,model)
    
    def decision_tree(self,X_train,y_train):
        from sklearn.tree import DecisionTreeRegressor
        
        criterion = ['mse', 'friedman_mse', 'mae']
        max_features = ["auto", "sqrt", "log2"]
        max_depth = np.arange(1,32)
        min_samples_split = np.linspace(0.1, 1.0, 9, endpoint=True)
        max_leaf_nodes = np.arange(2,20)
        self.parameters = dict(max_features = max_features,max_depth = max_depth,
                               min_samples_split =min_samples_split ,criterion = criterion,
                               max_leaf_nodes = max_leaf_nodes)
        
        model = DecisionTreeRegressor()
        self.get_best_model(X_train,y_train,model)

    def gradient_boosting(self,X_train,y_train):
        from sklearn.ensemble import GradientBoostingRegressor as GBR

        loss = ['ls','lad','huber','quantile']
        learning_rate = np.linspace(0.01,0.15,num = 10)
        n_estimators = np.arange(75,150)
        criterion = ['mse','mae','friedman_mse']
        max_depth = np.arange(3,10)
        subsample = np.linspace(0.5,1.5,10,endpoint=True)
        self.parameters = dict(loss=loss,learning_rate=learning_rate,n_estimators=n_estimators,criterion=criterion,max_depth=max_depth,subsample=subsample)
        model = GBR()
        self.get_best_model(X_train,y_train,model)

def regression_methods(ml_type,X_train,y_train,cv = None,path = 'model_path.txt'):
    
    """
    Description: Selecting classification method and apply it to the dataset
    
    Parameters:
        X_train,X_test,y_train,y_test: splitted datasets and corresponding labels

    """

    r = regressors(path,cv)
    machine_methods = {'SVM':r.SVR,'random_forest':r.random_forest,
                    'DeepNN':r.DNN,'decision_tree':r.decision_tree,
                    'gradient_boosting':r.gradient_boosting}   

    machine_methods[ml_type](X_train,y_train)

