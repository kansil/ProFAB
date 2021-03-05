# Benchmarking Platform Computational Protein Annotation Prediction Methods

This platform is generated to provide some machine learning algorithms like SVM, random forest etc. for the tasked dataset drug-target interaction, EC number prediction and GO ID prediction. This platform is based on supervised learning. 
Repository can be obtained by a single line of command:
```
git clone https://github.com/Sametle06/benchmark_platform.git
```

## Folders Description

This project is made to provide some platforms that includes some pre-processed datasets(without scaling) and training algorithms. The platforms are based on Drug-Target Interaction prediction (dtiPred), Enzyme Commission Number prediction(ecNoPred) and Gene Ontology ID prediction (goPred). These files include a lot of datasets from small size (n < 500) to big size (n > 200000) to train with a numerous learning algorithms. Also, to train these datasets, we provide easy to use machine learning algorithm. By defining the name of function, optimized and tuned results can be obtained. The links to the folders and simple descriptions are given below:

- **[platform_dtiPred](platform_dtiPred)**: includes run programs files and sample datasets for drug target interaction platform
- **[platform_ecNoPred](platform_ecNoPred)**: includes run programs files and sample datasets for ec number platform
- **[platfrom_goPred](platform_goPred)**: includes run programs files and sample datasets for GO ID platform
- **[scale_learn_evaluate](scale_learn_evaluate)**: is a folder that includes some file to scale, to train and to evaluate the perfomance of learnings.
- **[uniref_protein](uniref_protein)**: is a folder that contains information how to get UniRef Proteins. The data we get was uniref50-2020-05, download date was 20/11/2020.
- **[compound_featuring](compound_featuring)**: can be used to get rdkit.Bitvector of molecules.
- **to_fasta_**: To convert target sequence file to fasta format to get numerical features from iLearn web-tool(Chen, 2019).

## How to run the machine learning algorithms 

Before running the algorithms, it should be noted that the dataset folder which includes datasets needs to include some specific files. The name of these files are given in their section in details.

- **bio_main.py**: main file to run learning methods to train drug target interaction datasets. To run simply define the followning line as a command 
```
python bio_main.py --dataset_name folder_name
```

- **ec_main.py**: main file to run learning methods to train enzyme commission number datasets. To run simply define the followning line as a command
```
python ec_main.py --dataset_name file_name
```
- **go_main.py**: main file to run learning methods to train GO ID datasets. To run simply define the followning line as a command:
```
python go_main.py --dataset_name file_name
```

Explanation of parameters that are used to train models:
*    -**dataset_name**: folder that training model and scores are stored (user_determined)
*    -**scaler_type**:{'Standard_Scaler', 'Normalization', 'MinMax_Scaler', 'MaxAbs_Scaler', 'Robust_Scaler'}, (default: 'MinMax_Scaler'), It is used to scale the data to eleminate biases among the data
*    -**protein_feature**: {'paac', 'aac', 'gaac', 'eaac', 'ctriad', 'socnumber'}, (default: 'paac'), numerical feature of targets according to their sequences. If defined datasets do not come from these feature, please define the name of your feature and give a name to your dataset according to naming rule.  
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























