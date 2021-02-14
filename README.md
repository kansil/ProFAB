# Benchmarking Platform Computational Protein Annotation Prediction Methods

This platform is generated to provide some machine learning algorithms like SVM, random forest etc. for the tasked dataset drug-target interaction, EC number prediction and GO ID prediction. This platform is based on supervised learning. 
Repository can be obtained by a single line of command:
```
git clone https://github.com/Sametle06/benchmark_platform.git
```

## Folders Description

- **scale_learn_evaluate**: is a folder that contain scaler.py, classifications.py, regrassions.py and evaluation_metrics.py. These files are used to scale, to train and to check the trained model on test sets

- **platform_dtiPred**: is a folder to train bioactivity dataset that are taken from ChEMBLdb/Chembl_27. By training a regression model, bioactivity prediction btw compounds and targets(proteins) is done.

- **platform_ecNoPred**: is a folder to train enzyme comission number(EC No) dataset that are taken from UniProt/SwissProt. By training a classification model, EC No prediction is done. 

- **uniref_protein**: is a folder that contain only Uniref50 proteins.. It is used to separate the protein accoring to their similarities. Ours, uniref50 2020 05, download date was 20/11/2020 

- **bio_main.py**: a main file to run bioactivity dataset prediction main file. 
It can be done by a line of code (dataset_name = 'nr_data' for this example, others are default parameters):
```
python bio_main.py --dataset_name nr_data --split_type random_split --scaler_type MinMax_Scaler --learning_type Regression --machine_type random_forest --ratio 0.2 --cv None
```

Explanation of particular parameters for bioactivity dataset prediction
*    -**split_type**, {'random_split','compound_split','target_split','compound_target_split'}, (default:'random_split'): it is used to split data according features of compound and target

- **ec_main.py**: a main file to run EC number dataset prediction main file. 
It can be done by a line of code (dataset_name = '../EC_level_1/class_1' for this example, others are default parameters):
```
python ec_main.py --dataset_name ../EC_level_1/class_1 --split_type random --scaler_type MinMax_Scaler --learning_type Regression --machine_type random_forest --ratio 0.2 --cv None
```
Explanation of particular parameters for EC number dataset prediction
*    -**split_type**, {'random_split','target_split'}, (default:'random'): it is used to split data according features of compound and target
- **go_main.py**: a main file to run GO

Explanation of common parameters
*    -**dataset_name**: folder that training model and scores are stored (user_determined)
*    -**scaler_type**:{'Standard_Scaler','Normalization','MinMax_Scaler','MaxAbs_Scaler','Robust_Scaler'}, (default: 'MinMax_Scaler'), It is used to scale the data to eleminate biases among the data
*    -**learning_type**:{'Regression','Classification'}, (default: 'Regression'), to select which learning type will be used to train your data.
*    -**protein_feature**: {'paac',''}, (default: 'paac'), numerical ways to define protein sequences
*    -**machine_type**: 
        for regression: {'random_forest','SVR','DNN','decision_tree','gradient_boosting'},
   	    for classification:{'random_forest','SVM','DNN','KNN','naive_bayes,decision_tree',gradient_boosting}, 
   	    (default: 'random_forest(for both))', to choose which machine will be to train the dataset.
*    -**ratio**: Train test and validation sets split ratios. For ratio:0.2, 
                72% is train %10 test and %18 validation set 
*    -**cv**, (default: None): cross_validation which can be determined by user. If left None, RepeatedKFold() function will be applied to use in RandomizedSearch() function


#### Output of both main python files
    - **Model_Data**: saves the model of training to use in later datasets
    - **Score_Data**: saves the evaluation metrics of test and train sets

























