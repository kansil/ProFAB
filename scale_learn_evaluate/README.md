## Description

In this folder, there are machine learning, scaling and evaluation alogirthms. They are used in main() function to preprocess, predict and evaluate the data

### File Descriptions:

	- **scaler.py**: It is the module to apply scaler function and normalization to the dataset. All functions obtained by scikit-learn package of python. Functions are:
    	- Normalizer
    	- Standard_Scaler
    	- maxAbs_Scaler
    	- minMax_Scaler
    	- Robust_Scaler
    - **regressions.py**: It is the machine that is used to make regression on the dataset and to evalate it through some scores.Augmented machine learning algorithms are:
        - SVR(support vector machine)
        - random_forest
        -  DNN(deep neural network)
        - decision_tree
        - gradient_boosting
    - **classificaions.py**:  It is the machine that is used to make regression on the dataset and to evalate it through some scores.Augmented machine learning algorithms are:
        - logistic_regression
        - ridge_classifier
        - KNN(k-nearest neighbor)
        - SVM(support vector machine)
        - random_forest
        - DNN(deep neural network)
        - naive_bayes
        - decision_tree
        - gradient_boosting
    - **evaluate_metric**: It is the module that is used apply different evaluation metrics. These are:
		- mean squared error
		- root mean sqeared error
		- Pearson correlation coefficient
		- Spearman's rank correlation coefficient
		- precision
		- recall
		- f1
		- accuracy
		- Matthews correlation coefficient


