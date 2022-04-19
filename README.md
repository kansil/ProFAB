# ProFAB: Protein Functional Annotation Benchmark

ProFAB is a benchmarking platform for GO term and EC number prediction. It provides several datasets, featurization and scaling methods, machine learning algorithms and evaluation metrics. 

![mainFig_white](https://user-images.githubusercontent.com/37181660/164018357-1657bd95-ea13-4528-b878-15222525c143.svg)

As seen from the figure, in ProFAB, four main modules [import_dataset](profab/import_dataset), [model_preprocess](profab/model_preprocess), 
[model_learn](profab/model_learn), [model_evaluate](profab/model_evaluate) are employed.
- ***import_dataset***: lets to construct individual datasets for each GO term and EC number.
- ***model_preprocess***: provides three submodules for data preprocessing i.e., splitting, featurization and scaling.
- ***model_learn***: consists of several machine learning algorithms for binary classification. In this module, hyperparameter optimization is automatically done to determine the best performing models.
- ***model_evaluate***: provides several evaluation metrics to assess the performance of the trained models.

ProFAB availabilty:
    Operating System: Platform independent (except Protein Feature Extraction which can be run in LINUX and MAC.)\
    Programming language: Python: >=3.7\
    Package Requirements: tqdm (4.63.0), requests 2.27.1), numpy (1.21.2), scikit-learn (1.0.1)

To get repository, execute following line:
```
git clone https://github.com/Sametle06/benchmark_platform.git
```

## How to use ProFAB:

<br/>ProFAB has many workloads, therefore, reading the introductions is highly recommended. Detailed explanations can be found in each module: [import_dataset](profab/import_dataset), [model_preprocess](profab/model_preprocess), [model_learn](profab/model_learn), [model_evaluate](profab/model_evaluate).

To run ProFAB on terminal, [easy_profab](easy_profab.py) can be used. Its parameters are given in Table.1:

Table.1: Parameters to run ProFAB on terminal:
    
| Parameters (type) | options | default | Definition                                      |
|:-------------:|:-----------------:|:---------:|:---------|
file_name (str)|-|-| File includes dataset names such as GO_0000018, GO_1905523. If *isUser* = True or *isFasta* = True, then directory to dataset folder must be defined in input file. Each must be defined in new line
score_path (str)|-|'score_path.csv'| A destination where scores are saved. It must be .csv file
model_path (str)|-|None| A destination where model parameters of given dataset are saved. 
set_type (str)| 'random'<br/>'similarity'<br/>'temporal'| 'random'| split type of data, random: random splitting, target: similarity based splitting, temporal: splitting according to annotation time. If *isUser* or *isFasta* is True, random splitting will be applied to data even though set_type is not 'random' splitting. 'similarity' and 'temporal' splitting options are valid for only ProFAB datasets.
protein_feature (str)| 'paac'<br/>'aac'<br/>'gaac'<br/>'ctriad'<br/>'ctdt'<br/>'soc_number'<br/>'kpssm' | 'paac'| numerical features of protein sequences. If *isFasta* = True, options can be found in [Table.2 and Table.3](profab/utils/feature_extraction_module/README.md) 
ratio (float, list)| - | 0.2 | used to split data into train, test, validation sets as given values. If ratio = a (float), then test will be a% of total data size. If ratio = [a,b] where a and b are in (0,1), train, test and validation sets are formed according to them. For example, If a = 0.2 and b = 0.1, train fraction is 0.7, test fraction is 0.2 and validation fraction is 0.1 of all dataset size. If set_type = 'temporal', then ratio = None is set automatically by ProFAB.
pre_determined (bool)| - | False | if False, data is given according to ratio type, If True, already splitted data will provided
scale_type (str)| 'normalizer'<br/>'standard'<br/>'max_abs'<br/>'min_max'<br/>'robust'|'standard' |determines the method to scale the data
ml_type (str) | 'logistic_reg'<br/>'ridge_class'<br/>'KNN'<br/>'SVM'<br/>'random_forest'<br/>'MLP'<br/>'naive_bayes'<br/>decision_tree'<br/>'gradient_boosting' |'logistic_reg'| type of machine learning algorithm
isFasta (bool) | - |False| If True, a data provided by user is Fasta file else numerical data should be introduced. While *isUser* = True, this parameter cannot be True at the same time. Format of fasta files must be **.fasta** and names of files should describe label. The path described in input file must include these files: "positive_data.fasta" and "negative_data.fasta"
place_protein_id (int)| - | 1 | It indicates the place of protein id in fasta header. e.g. fasta header: >sp|O27002|....|....|...., seperate the header wrt. '|' then >sp is in the zeroth position, protein id in the first(1) position
isUser (bool)| - | False| If True, user data path must be defined in input file. While *isFasta* = True, this parameter cannot be True at the same time. If *label* = False, names of files should describe label. As an example, The path described in input file must include these files: "positive_data.txt" and "negative_data.txt". If ***label*** = True, it doesn't matter
delimiter (str)| '\t' (tab)<br/>',' (comma)<br/>' ' (space)|'\t'| a character to separate columns in file
name (bool)| - |False| If True, then first colmun is considered as name of inputs else the first column is a feature column
label (bool)| - | False| If True, then last colmun is considered as label of inputs else the last column is a feature column 

<br/>It can be run on terminal with a single line:

where *isFasta* = False and *isUser* = False, use support vector machine as training algorithms and save perfomance
of model to *my_score_path.csv*:
```{python}
python easy_profab.py --file_name sample_inputs.txt --score_path my_score_path.csv --ml_type SVM
```
where *isUser* = True, use k-nearest neighbor as training algorithm and test fraction is 0.3 and feature matrices include names of instances:
```{python}
python easy_profab.py --file_name sample_inputs_userTrue.txt --isUser True --ml_type KNN --ratio 0.3 --name True
```
where *isFasta* = True, use random forest as training algorithm , protein descriptor is CTRIAD, test fraction is 0.1 & validation fraction is 0.2:
```{python}
python easy_profab.py --file_name sample_inputs_fastaTrue.txt --isFasta True --ml_type random_forest --protein_feature CTriad --ratio 0.1,0.2
```

<br/>ProFAB can be run in pythonic way. How to apply its functions are shown in two different use cases. [use_case_1](use_case/use_case_1.ipynb) is based on utilizing ProFAB datasets whereas [use_case_2](use_case/use_case_2.ipynb) is based on integrating user itself datasets. Detailed explanations can be found in links.


## License

MIT License

ProFab Copyright (C) 2020 CanSyL

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
