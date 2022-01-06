# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:45:38 2020

@author: Sameitos
"""


import numpy as np
from sklearn.model_selection import RandomizedSearchCV,RepeatedStratifiedKFold, PredefinedSplit
import pickle

import warnings
warnings.filterwarnings("ignore")


class classifiers(object):

    
    def __init__(self,path):     
        """

        Description: In class,6 different machine learning methods for regression 
                are introduced. Their hyperparameters are tuned by
                RandomizedSearchCV and all methods return only their hyperparameters 
                that give the best accoring to cvthat is created by RepeatedStraitKFold.
    
        Parameters:
            path: where outcome of training is saved    
        """
        self.path = path
        self.parameters = None
        self.n_jobs = -1
        self.random_state = 0

      
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
            
            cv = RepeatedStratifiedKFold(n_splits=10,n_repeats = 5,random_state= self.random_state)
            
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
                                     scoring="f1", random_state = self.random_state)

        if y_train is not None:
            clf.fit(X_train,y_train)
        else:
            clf.fit(X_train)

        best_model = clf.best_estimator_
        print(best_model)
        if self.path is not None:
	        with open(self.path, 'wb') as f:
	            pickle.dump(best_model,f)

        return best_model


    def logistic_regression(self,X_train,y_train,X_valid,y_valid):    
        from sklearn.linear_model import LogisticRegression
        from ..parameters import cls_logistic_regression_params as lrp

        self.parameters = lrp
        model = LogisticRegression()
        return self.get_best_model(model, X_train, y_train,X_valid, y_valid)
        

    def ridge_class(self,X_train,y_train,X_valid,y_valid):
        from sklearn.linear_model import RidgeClassifier
        from ..parameters import cls_ridge_class_params as rcp

        self.parameters = rcp
        model = RidgeClassifier()
        return self.get_best_model(model, X_train, y_train,X_valid, y_valid)

        
    def KNN(self,X_train,y_train,X_valid,y_valid):
        from sklearn.neighbors import KNeighborsClassifier
        from ..parameters import cls_knn_params as kp

        self.parameters  = kp
        model = KNeighborsClassifier()
        return self.get_best_model(model, X_train, y_train,X_valid, y_valid)

    
    def SVM(self,X_train,y_train,X_valid,y_valid):
        from sklearn.svm import SVC
        from ..parameters import cls_svm_params as sp
        
        self.parameters = sp
        model = SVC()
        return  self.get_best_model(model, X_train, y_train,X_valid, y_valid)


    def random_forest(self,X_train,y_train,X_valid,y_valid):
        from sklearn.ensemble import RandomForestClassifier
        from ..parameters import cls_random_forest_params as rfp
        
        self.parameters = rfp
        model = RandomForestClassifier()   
        return  self.get_best_model(model, X_train, y_train,X_valid, y_valid)


    def MLP(self,X_train,y_train,X_valid,y_valid):
        from sklearn.neural_network import MLPClassifier
        from ..parameters import cls_mlp_params as mlpp

        self.parameters = mlpp
        model = MLPClassifier()
        return self.get_best_model(model, X_train, y_train,X_valid, y_valid)


    def decision_tree(self,X_train,y_train,X_valid,y_valid):
        from sklearn.tree import DecisionTreeClassifier
        from ..parameters import cls_decision_tree as dtp

        self.parameters = dtp
        model = DecisionTreeClassifier()
        return self.get_best_model(model, X_train, y_train,X_valid, y_valid)


    def gradient_boosting(self,X_train,y_train,X_valid,y_valid):
        from sklearn.ensemble import GradientBoostingClassifier as GBC
        from ..parameters import cls_gradient_boosting as gbp
        
        self.parameters = gbp
        model = GBC()
        return self.get_best_model(model, X_train, y_train,X_valid, y_valid)


    def naive_bayes(self,X_train,y_train,X_valid,y_valid):
        from sklearn.naive_bayes import GaussianNB
        
        model = GaussianNB()
        model.fit(X_train,y_train)
        with open(self.path,'wb') as f:
            pickle.dump(model, f)
        return model


def classification_methods(ml_type,X_train,y_train = None ,
                           X_valid = None,y_valid = None,
                           path = None):
    
    """
    Description: Selecting classification method and apply it to the dataset
    
    Parameters:
        X_train,X_valid,y_train,y_valid: splitted datasets and corresponding labels
    """
    if not set(y_train) == {1,-1} or set(y_train) == {1,0}:
    	raise ValueError(f'Data must be binary: {{1,-1}} or {{1,0}}')

    c = classifiers(path)
    
    machine_methods = {'logistic_reg':c.logistic_regression,'ridge_class':c.ridge_class,
                     'KNN':c.KNN,'SVM':c.SVM,'random_forest':c.random_forest,
                    'MLP':c.MLP,'naive_bayes':c.naive_bayes,'decision_tree':c.decision_tree,
                    'gradient_boosting':c.gradient_boosting}    

    return machine_methods[ml_type](X_train = X_train,
                                    y_train = y_train,
                                    X_valid = X_valid,
                                    y_valid = y_valid)





















