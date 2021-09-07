# Benchmarking Platform for Computational Protein Annotation Prediction

This platform is generated to provide some machine learning algorithms like SVM, random forest etc. for the tasked dataset drug-target interaction, EC number prediction and GO ID prediction. This platform is based on supervised learning. 
Repository can be obtained by a single line of command:
```
git clone https://github.com/Sametle06/benchmark_platform.git
```

## Folders Description

This project is made to provide some platforms that includes some pre-processed datasets(without scaling) and training algorithms. The platforms are based on Drug-Target Interaction prediction (dtiPred), Enzyme Commission Number prediction(ecNoPred) and Gene Ontology ID prediction (goPred). These files include a lot of datasets from small size (n < 500) to big size (n > 200000) to train with a numerous learning algorithms. Also, to train these datasets, we provide easy to use machine learning algorithm. By defining the name of function, optimized and tuned results can be obtained.

## How to run the machine learning algorithms 

Training algorithms can be used in two ways. If the user has his/her dataset, s/he can use the learning algorithm by simply defining the following line to the terminal:
```
python train_self.py --dataset_name exp_folder_name
``` 
All other parameters have their default values. They can be also changed by defining the values in the line. The description of the paramters:

*    -**dataset_name**: folder where user datasets are stored in here. Also, in this folder outputs that aretraining model and scores will be stored.
*    -**scaler_type**:{'Standard_Scaler', 'Normalization', 'MinMax_Scaler', 'MaxAbs_Scaler', 'Robust_Scaler'}, (default: 'MinMax_Scaler'), It is used to scale the data to eleminate biases among the data
*	 -**learning_type**: {'Regression,'Classification}, (default: Classification), It will be used to indicate with learining method will be used to train and the evaluate the score.
*    -**machine_type**: for regression: {'random_forest','SVR','DNN','decision_tree','gradient_boosting'}, for classification:{'random_forest','SVM','DNN','KNN','naive_bayes,decision_tree',gradient_boosting}, 
   	    (default: 'random_forest(for both))', to choose which machine will be to train the dataset.
*    -**ratio**: Train test and validation sets split ratios. If float, train and test sets will be formed,
            If list with size = 2, ratio = [test_ratio,validation_ratio], train, test and validation sets
            will be formed. If datasets are already diveded in train and test, 
            ratio must be None, or float. If float, validation set will be formed via train set. 
            If None, train and test data will be used to train and calculate scores. (default = 0.2)
*    -**cv**, (default: None): cross_validation which can be determined by user. If left None, RepeatedKFold() function will be applied for tuning.

However, to use this way, the user has to define some files in a folder before the assignment. The name of the file is given in this section in detail: [import_datasets](profab/import_dataset).
The output of this methods are:
```
Model_file: Model_machine_type.txt
Score_file: Score_machine_type.csv
```

The other way to use the learning algorithms is passing from using any Python IDE by importing the packages. The way to implement the code is given in [test_file](test_file.ipynb).

The parameters used in dataset importing are explained in [import_datasets](profab/import_dataset). Other steps are explained in [model_process](probab/model_process) and in [model_evaluate](profab/model_evaluate).

## Compound Featurization

For the users who want to obtain trainable data from molecules, we provides a program that converts SMILES data of compound to rdkit.BitVector. For a clear explanation visit [compound_featuring](profab/compound_featuring). The program can be run with a line of command:
```
python compound_to_ --data_name smiles.txt --save_data feature.txt --save_idx compound_indices.txt --bits 1024
```
Output of the command: 
```
- feature.txt: contains Chembl Molecules and BitVectors in string by rdkit.
- indices.txt: contains indices of original data.
```
