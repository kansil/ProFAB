## Description

Several machine learning algprithms based on regression and classification are available. Tuning hyperparameters is achieved with randomized search. The hyperparameters of each algorithm are portrayed in [hyperparameters.py](hyperparameters.py). These hyperparameters can be redefined to improve performance of models.

## Methods

### Binary Classifications

After the preprocessing steps (i.e., obtaining the dataset, featurization and scaling), below classification methods can be used for binary classification. All algorithms are based on scikit-learn Python package.

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

- ***X_train***: type = {list, numpy array} feature matrix, train set
- ***X_valid***: default = None, feature matrix validation set
- ***y_train***: default = None, label matrix, train set
- ***y_valid***: default = None, label matrix validation set
- ***ml_type***: ml_type: {'logistic_reg','ridge_class','KNN','SVM','random_forest','MLP',
                'naive_bayes', decision_tree',gradient_boosting'}, default = "SVM",
                Type of machine learning algorithm.
- ***path***: default = None, A destination point where model is saved

#### Usage

```{python}
from profab.model_learn import classification_methods
model = classification_methods(ml_type,
                                X_train,
                                y_train,
                                X_valid,
                                y_valid,
                                path)
```

A use case:
```{python}
from profab.model_learn import classification_methods
model = classification_methods(ml_type = 'logistic_reg',
                                X_train = X_train,
                                y_train = y_train)
```

### Regression

ProFAB also provides machine learning algorithms for regression to estimate continous outputs. For now, no data is available for regression task in ProFAB datasets. As like classification, all algorithms are based on python package scikit-learn. Used algorithms are:

    - SVR(support vector machine)
    - random_forest
    - DNN(deep neural network)
    - decision_tree
    - gradient_boostin

#### Explanation of Parameters

- ***X_train***: type = {list, numpy array} feature matrix, train set
- ***X_valid***: default = None, feature matrix validation set
- ***y_train***: default = None, label matrix, train set
- ***y_valid***: default = None, label matrix validation set
- ***ml_type***: {'linear_reg','SVM','random_forest','MLP',
                'naive_bayes', decision_tree',gradient_boosting'}, default = "SVM",
                Type of machine learning algorithm.
- ***path***: default = None, A destination point where model is saved

#### Usage

Function usage is the same with "classification_methods".

```{python}
from profab.model_learn import regression_methods
model = regression_methods(ml_type,
                                X_train,
                                y_train,
                                X_valid,
                                y_valid,
                                path)
```
