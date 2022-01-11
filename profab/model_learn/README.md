## Description

Model learn includes machine learning algprithms based on regression and classification and ProFAB provides tuning hyperparameters with randomized search. The hyperparameters of each algorithm are portrayed in [hyperparameters.py](../hyperparameters.py). These hyperparameters can be redefined to increase performance of models.

## Methods

### Binary Classifications

Binary classification is main task of ProFAB which is used to classify proteins according to their proteins. To achieve that different methods based on python scikit-learn package are presented. Used algorithms are:

    - logistic_regression
    - ridge_classifier
    - KNN(k-nearest neighbor)
    - SVM(support vector machine)
    - random_forest
    - DNN(deep neural network)
    - naive_bayes
    - decision_tree
    - gradient_boosting

#### Explanation of Parameters

-**X_train**: type = {list, numpy array} feature matrix, train set
-**X_valid**: default = None, feature matrix validation set
-**y_train**: default = None, label matrix, train set
-**y_valid**: default = None, label matrix validation set
-**ml_type**: default = "SVM", type of machine learning algorithm
-**path**: default = None, A destination point where model is saved

#### Usage

```{python}
from profab.model_learn import classification_methods
model = classification_methods(ml_type,
                                X_train,
                                y_train,
                                X_valid,
                                y_valid)
```

A use case:
```{python}
from profab.model_learn import classification_methods
model = classification_methods(ml_type = 'logistic_reg',
                                X_train = X_train,
                                y_train = y_train)
```

### Regression

Regression was prepared to estimate continous outputs. In ProFAB, no such data is available. As like classification, all algorithms are based on python package scikit-learn. Used algorithms are:

    - SVR(support vector machine)
    - random_forest
    - DNN(deep neural network)
    - decision_tree
    - gradient_boostin

#### Explanation of Parameters

-**X_train**: type = {list, numpy array} feature matrix, train set
-**X_valid**: default = None, feature matrix validation set
-**y_train**: default = None, label matrix, train set
-**y_valid**: default = None, label matrix validation set
-**ml_type**: default = "SVM", type of machine learning algorithm
-**path**: default = None, A destination point where model is saved

#### Usage

```{python}
from profab.model_learn import regression_methods
model = regression_methods(ml_type,
                                X_train,
                                y_train,
                                X_valid,
                                y_valid)
```
