# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:39:12 2020

@author: Sameitos
"""


import numpy as np
from sklearn.model_selection import RandomizedSearchCV,RepeatedKFold, PredefinedSplit
import pickle

import warnings
warnings.filterwarnings("ignore")





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
      
    
    def get_best_model(self, model, X_train, y_train,X_valid, y_valid):
        """
        Parameters
        ----------
        X_train: Feature matrix, type = {list, numpy array}
        y_train: (default = None), Label matrix, type = {list, numpy array}
        X_valid: (default = None), Validation Set, type = {list,numpy array}
        y_valid: (default = None), Validation Label, type = {list,numpy array}
        model : model specified in ML algorithms, type = string
        """

        if X_valid is None: 
            
            cv = RepeatedKFold(n_splits=10,n_repeats = 5,random_state= 2)
            
        else:
            
            if y_valid is None:
                raise ValueError(f'True label data for validation set cannot be None')
            
            X_train = list(X_train)
            y_train = list(y_train)
            
            len_tra, len_val = len(X_train),len(X_valid)
            
            X_train.extend(list(X_valid))
            y_train.extend(list(y_valid))
            
            test_fold = [0 if x in np.arange(len_tra) else -1 for x in np.arange(len_tra+len_val)]
            cv = PredefinedSplit(test_fold)


        clf = RandomizedSearchCV(model,self.parameters,n_iter = 10,
                                     n_jobs=self.n_jobs, cv = cv,
                                     scoring="f1")

        
        if y_train is not None:
            clf.fit(X_train,y_train)
        else:
            clf.fit(X_train)
        best_model = clf.best_estimator_
        if self.path is not None:
            with open(self.path, 'wb') as f:
                pickle.dump(best_model,f)
        return best_model


    def linear_regression(self,X_train,y_train,X_valid,y_valid):

        from sklearn.linear_model import LinearRegression
        from ..parameters import rgr_linear_regression_params as lrp
        
        self.parameters = lrp
        model = LinearRegression()
        return self.get_best_model(model,X_train,y_train,X_valid,y_valid)

    def SVM(self,X_train,y_train,X_valid,y_valid):
        from sklearn.svm import SVR
        from ..parameters import rgr_svm_params as svmp
        
        self.parameters = svmp
        model = SVR()
        
        return self.get_best_model(model, X_train, y_train,X_valid, y_valid)
        
    def random_forest(self,X_train,y_train,X_valid,y_valid):
        from sklearn.ensemble import RandomForestRegressor
        from ..parameters import rgr_random_forest_params as rfp
        
        self.parameters = rfp
        model = RandomForestRegressor()   
        return self.get_best_model(model, X_train, y_train,X_valid, y_valid)

    def MLP(self,X_train,y_train,X_valid,y_valid):
        
        from sklearn.neural_network import MLPRegressor
        from ..parameters import rgr_mlp_params as mlpp
        
        self.parameters = mlpp
        model = MLPRegressor()
        return self.get_best_model(model, X_train, y_train,X_valid, y_valid)
    
    def decision_tree(self,X_train,y_train,X_valid,y_valid):
        from sklearn.tree import DecisionTreeRegressor
        from ..parameters import rgr_decision_tree_params as dtp
        
        self.parameters = dtp
        model = DecisionTreeRegressor()
        return self.get_best_model(model, X_train, y_train,X_valid, y_valid)

    def gradient_boosting(self,X_train,y_train,X_valid,y_valid):
        from sklearn.ensemble import GradientBoostingRegressor as GBR
        from ..parameters import rgr_gradient_boosting_params as gbp

        self.parameters = gbp
        model = GBR()
        return self.get_best_model(model, X_train, y_train,X_valid, y_valid)


def regression_methods(ml_type,X_train,y_train = None ,X_valid = None,y_valid = None, path = None):
    
    """
    Description: Selecting classification method and apply it to the dataset
    
    Parameters:
        X_train,X_test,y_train,y_test: splitted datasets and corresponding labels

    """

    r = regressors(path)
    
    machine_methods = {
                        'linear_regression':linear_regression,
                        'SVM':r.SVM,
                        'random_forest':r.random_forest,
                        'DeepNN':r.MLP,
                        'decision_tree':r.decision_tree,
                        'gradient_boosting':r.gradient_boosting
                    }   

    machine_methods[ml_type](X_train = X_train,y_train = y_train,X_valid = X_valid,y_valid = y_valid)

    return path



















