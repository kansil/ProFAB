## Description

Model_evaluate module is to predict test labels, calculating scores of models performance and tabularizing them. For regression and classification different scoring metrics are presented to observe performance in many aspects.

### Evaluation

This module is to considering performance results with many scoring metrics. Used metrics are

    - Regression:
        - mean squared error
        - root mean sqeared error
        - Pearson correlation coefficient
        - Spearman's rank correlation coefficient
        - Threshold based classification Metrics
    - Classification:
        - precision
        - recall
        - f1
        - f0.5
        - accuracy
        - Matthews correlation coefficient
        - AUROC
        - AUPRC

#### Explanation of Parameters

-**model**: Fitting to predict labels
-**X**: type = {list, numpy array}, feature matrix to introduce to model
-**y**: type = {list, numpy array}, corresponding label matrix
-**preds**: type = bool, If True return predictions and scores else only return scores
-**learning_method**: {"classif","reg"}, default = "classif", Learning task to get corresponding metrics

#### Usage

```{python}
from profab.model_evaluate import evaluate_score
return_value = evaluate_score(model,
                            X,
                            y,
                            preds)
```

A use case:
```
python 

score_test,f_test = evaluate_score(model,
                                X = X_test, 
                                y = y_test, 
                                preds = True)

```

### Tabularizing the Metrics

To see scores in .csv files in an order, this function is proposed.

#### Explanation of Parameters

--**scores**: type = {dict}, includes scores of sets (train, test)
--**learning_method**: {"classif","reg"}, default = "classif", to set values in order
--**path**: default = 'score_path.csv', destination where table will be saved. Format must be .csv

#### Usage

A use case:
```
python

from profab.model_evaluate import form_table

scores = {'test':score_test}

form_table(scores = scores)
```

