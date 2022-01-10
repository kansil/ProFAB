# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 13:06:23 2021

@author: Sameitos
"""

import numpy as np

'''
    This file to modify the parameters of machine learning algorithms w/o 
    meddling with the main functions. One can change, delete or add parameters
    with their true names. All parameters are provided for two sections "Regression" 
    and "Classification".
'''



'''
Classification Parameters for algorithms:
            logistic regression
            ridge class
            k-nearest neighbor
            support vector machine
            random_forest
            multi layer perceptron
            decision tree
            gradient boosting
'''
#Logistic Regression Parameters
cls_logistic_regression_params = dict(
            penalty = ['l2'],
            multi_class = ["auto",'ovr'],
            solver = ['newton-cg','saga', 'sag', 'lbfgs', 'liblinear'],
            C = np.linspace(0.001,100, num=100),
            max_iter = [2000]
            )

#Ridge Class Parameters
cls_ridge_class_params = dict(
            alpha = np.linspace(0.1,1,num=1000),
            fit_intercept = [False,True],
            normalize = [False,True]
            )

#k-Nearest Neighbor Parameters
cls_knn_params = dict(
            n_neighbors = np.arange(1,15),
            weights = ['uniform', 'distance'],
            metric = ['euclidean', 'manhattan', 'minkowski'],
            leaf_size = np.linspace(10,100,num = 20),
            algorithm = ['ball_tree', 'kd_tree']
            )

#Support Vector Machine Parameters
cls_svm_params = dict(
            kernel = ["linear","poly","rbf","sigmoid"],
            C = np.linspace(0.5,50,200),
            gamma = [10**x for x in np.linspace(-3,0,num=50)],
            max_iter = [2500]
            )

#Random Forest Parameters
cls_random_forest_params = dict(
            n_estimators = [int(i) for i in np.linspace(10,50,num=40)],
            max_features = ["auto","sqrt","log2"],
            bootstrap = [True, False],
            min_samples_split = np.linspace(0.09, 1.0, 50, endpoint=True),
            ccp_alpha = [10**x for x in np.linspace(-3.1,-1.5,num=100)]
            )

#Multilayer Perceptron Parameters
cls_mlp_params = dict(
            hidden_layer_sizes = ([int(x) for x in np.arange(100,300)],[int(x) for x in np.arange(1,5)]),
            activation = ['logistic', 'tanh', 'relu'],
            solver = ['sgd','adam'],
            alpha = np.linspace(0.0001,0.001,num = 10),
            batch_size = [100,150],
            learning_rate = ['constant', 'invscaling', 'adaptive'],
            max_iter = [1000]
            )

#Decision Tree Parameters
cls_decision_tree =dict(
            max_features = ["auto", "sqrt", "log2"],
            #max_depth = np.arange(1,32),
            #min_samples_leaf = np.arange(10,50),
            criterion = ['gini','entropy'],
            ccp_alpha = [10**x for x in np.linspace(-3.5,-1.5,num=50)]
            )

#Gradient Boosting Parameters
cls_gradient_boosting = dict(
            loss = ['deviance','exponential'],
            learning_rate = np.linspace(0.01,0.15,num = 10),
            n_estimators = np.arange(100,150),
            criterion = ['mse','friedman_mse'],
            #max_depth = np.arange(3,10),
            ccp_alpha = [10**x for x in np.linspace(-3.4,-1.5,num=50)]
            )
            

'''
Regression parameters for algorithms:
            linear regression
            support vector machine
            random forest
            multilayer perceptron
            decision tree
            gradient boosting
'''
#Linear Regression Parameters
rgr_linear_regression_params = dict(
            fit_intercept = True,
            positive = True
            )

#Support Vector Machine Parameters
rgr_svm_params = dict(
            kernel = ["poly","rbf","sigmoid",'sigmoid'],
            C = np.linspace(0.1,50,100),
            gamma = [10**x for x in np.arange(-5,5,dtype = float)],
            max_iter = np.arange(50,200)
            )

#Random Forest Parameters
rgr_random_forest_params = dict(
            n_estimators = [int(i) for i in np.linspace(10,50,num=40)],
            max_features = ["auto","sqrt","log2"],
            bootstrap = [True, False],
            max_depth = np.arange(1,32),
            min_samples_split = np.linspace(0.1, 1.0, 9, endpoint=True)
            )

#Multilayer Perceptron Parameters
rgr_mlp_params = dict(
            hidden_layer_sizes = ([int(x) for x in np.arange(100,300)],[int(x) for x in np.arange(1,5)]),
            activation = ['logistic', 'tanh', 'relu'],
            solver = ['sgd','adam'],
            alpha = np.linspace(0.0001,0.001,num = 10),
            batch_size = [100,150],
            learning_rate = ['constant', 'invscaling', 'adaptive'],
            max_iter = np.arange(20,50)
            )

#Decision Tree Parameters
rgr_decision_tree_params = dict(
            criterion = ['mse', 'friedman_mse', 'mae'],
            max_features = ["auto", "sqrt", "log2"],
            max_depth = np.arange(1,32),
            min_samples_split = np.linspace(0.1, 1.0, 9, endpoint=True),
            max_leaf_nodes = np.arange(2,20)
            )

#Gradient Boosting Paramters
rgr_gradient_boosting_params = dict(
            loss = ['ls','lad','huber','quantile'],
            learning_rate = np.linspace(0.01,0.15,num = 10),
            n_estimators = np.arange(75,150),
            criterion = ['mse','mae','friedman_mse'],
            max_depth = np.arange(3,10),
            subsample = np.linspace(0.5,1.5,10,endpoint=True)
            )

























