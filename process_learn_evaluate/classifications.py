# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:45:38 2020

@author: Sameitos
"""

# This program is used for binary classification({-1,1}) for proteins in different location of cells
# to achieve that, machine learning algorithms from sklearn lib were used.

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV,RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
import pickle
from pathlib import Path
import os
    
class classifiers(object):
    """
    Description: In class,6 different machine learning methods for regression are introduced. Their hyperparameters are tuned by
                RandomizedSearchCV and all methods return only their hyperparameters that give the best accoring to cv
                that is created by RepeatedStraitKFold.
    
    Parameters
    ----------
        X_train: feature matrix
        y_train: label matrix
       : path that model will be saved
                
    """
    
    def __init__(self,path,ml_type,cv = None):
        
        """
        ------------
        parameters: Traning model parameters
        cv: repeated K-Fold Cross Validation 
        ------------

        """
        self.path = path
        self.ml_type = ml_type 
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
            self.cv = RepeatedStratifiedKFold(n_splits=10,n_repeats = 5,random_state= 2)
            clf = RandomizedSearchCV(model,self.parameters,
                                     cv= self.cv, 
                                     n_jobs=self.n_jobs,
                                     scoring="neg_mean_squared_error")
        else: 
            clf = RandomizedSearchCV(model,self.parameters,
                                     n_jobs=self.n_jobs,
                                     scoring="neg_mean_squared_error")

        clf.fit(X_train,y_train)
        best_model = clf.best_estimator_
        with open(self.path, 'wb') as f:
            pickle.dump(best_model,f)

    def logistic_regression(self,X_train,y_train):    
        from sklearn.linear_model import LogisticRegression
        
        penalty = ['l1','l2']
        multi_class = ["auto",'ovr','multinomial']
        solver = ['newton-cg', 'lbfgs', 'liblinear']
        c = np.linspace(0.001,100, num=100)
        max_iter = np.arange(100,1000)
        self.parameters = dict(penalty = penalty,C=c,multi_class=multi_class,
                          max_iter = max_iter,solver = solver)
        model = LogisticRegression()
        self.get_best_model(X_train,y_train,model)
        
    def ridge_class(self,X_train,y_train):
        from sklearn.linear_model import RidgeClassifier
        
        alpha = np.linspace(0.1,1,num=1000)
        fit_intercept = [False,True]
        normalize = [False,True]
        self.parameters = dict(alpha = alpha,fit_intercept = fit_intercept,
                          normalize = normalize)
        model = RidgeClassifier()
        self.get_best_model(X_train,y_train,model)

        
    def KNN(self,X_train,y_train):
        from sklearn.neighbors import KNeighborsClassifier
        
        neighbors = np.arange(5,15)
        weights = ['uniform', 'distance']
        metric = ['euclidean', 'manhattan', 'minkowski']
        leaf_size = np.linspace(10,100,num = 20)
        algorithm = ['auto', 'ball_tree', 'kd_tree']
        self.parameters = dict(n_neighbors = neighbors, weights=weights,
                          metric=metric,leaf_size = leaf_size,algorithm=algorithm)
                          
        model = KNeighborsClassifier()
        self.get_best_model(X_train,y_train,model)
        
    
    def SVM(self,X_train,y_train):
        from sklearn.svm import NuSVC
        
        kernel = ["poly","rbf","sigmoid"]

        nu = np.linspace(0.23,1,200)
        gamma = [10**x for x in np.linspace(-3,-1,num=15)]
        max_iter = np.arange(1500,2000)
        self.parameters = dict(kernel = kernel,nu=nu,gamma = gamma,max_iter = max_iter,verbos = True)
        model = NuSVC()
        self.get_best_model(X_train,y_train,model)
        
    def random_forest(self,X_train,y_train):
        from sklearn.ensemble import RandomForestClassifier
        
        n_estimators = [int(i) for i in np.linspace(10,50,num=40)]
        max_features = ["auto","sqrt","log2"]
        bootstrap = [True, False]
        min_samples_split = np.linspace(0.1, 1.0, 10, endpoint=True)
        self.parameters = dict(n_estimators = n_estimators,max_features = max_features,
                          min_samples_split = min_samples_split,
                          bootstrap = bootstrap)
        model = RandomForestClassifier()   
        self.get_best_model(X_train,y_train,model)

    def DNN(self,X_train,y_train):
        from sklearn.neural_network import MLPClassifier
        
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
        model = MLPClassifier()
        self.get_best_model(X_train,y_train,model)
    
    def naive_bayes(self,X_train,y_train):
        from sklearn.naive_bayes import GaussianNB
        
        model = GaussianNB()
        model.fit(X_train,y_train)
        best_model = model
        with open(self.path,'w') as f:
            pickle.dump(best_model, f)

    def decision_tree(self,X_train,y_train):
        from sklearn.tree import DecisionTreeClassifier
        max_features = ["auto", "sqrt", "log2"]
        #max_depth = np.arange(1,32)
        criterion = ['gini','entropy']
        self.parameters = dict(max_features = max_features,#max_depth = max_depth,
                               criterion = criterion)
        model = DecisionTreeClassifier()
        self.get_best_model(X_train,y_train,model)
    
    def gradient_boosting(self,X_train,y_train):
        from sklearn.ensemble import GradientBoostingClassifier as GBC

        loss = ['deviance','exponential']
        learning_rate = np.linspace(0.01,0.15,num = 10)
        n_estimators = np.arange(100,170)
        criterion = ['mse','mae','friedman_mse']
        max_depth = np.arange(3,10)
        self.parameters = dict(loss=loss,learning_rate=learning_rate,n_estimators=n_estimators,criterion=criterion,max_depth=max_depth)
        model = GBC()
        self.get_best_model(X_train,y_train,model)

def classification_methods(path, ml_type,X_train,y_train,cv = None):
    
    """
    Description: Selecting classification method and apply it to the dataset
    
    Parameters:
        X_train,X_test,y_train,y_test: splitted datasets and corresponding labels
    
    Return : 
        Scores: F1,MCC,Precision, Recall, Accuracy, F0.5
    """
        
    c = classifiers(path,ml_type,cv)
    
    machine_methods = {'log_reg':c.logistic_regression,'ridge_class':c.ridge_class,
                     'KNN':c.KNN,'SVM':c.SVM,'random_forest':c.random_forest,
                    'DeepNN':c.DNN,'naive_bayes':c.naive_bayes,'decision_tree':c.decision_tree,
                    'GradientBoosting':c.gradient_boosting}    

    
    machine_methods[ml_type](X_train,y_train)