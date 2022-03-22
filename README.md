# ProFAB: Protein Functional Annotation Benchmark

This platform is generated to provide datasets amd shallow machine learning algorithms such as SVM, random forest etc. for protein function prediction. Its learning method is supervised and to present complete set of machine learning study preprocessing and evaluation steps are added, too. ProFAB's main purpose is for protein based studies but its functions can be used separately for different reasons. To illustrate, one can use only to import and splitting data while another one can use only machine learning algorithms thank to its flexible structure.

![mainFigNew](https://user-images.githubusercontent.com/37181660/150197153-9ce060d5-f0f5-4e9b-bcb5-2044173138da.png)

As seen from the figure, in ProFAB, four main modules [import_dataset](profab/import_dataset), [model_preprocess](profab/model_preprocess), 
[model_learn](profab/model_learn), [model_evaluate](profab/model_evaluate) are employed.
- ***import_dataset***: To load dataset from the files. Users can use ProFAB datasets or can load their data with related functions.
- ***model_preprocess***: To feed the data to learning algorithms, this module provides there pre-works:
	- featurization of protein sequence data
	- random splitting
	- and scaling
- ***model_learn***: To apply machine learning algorithms
- ***model_evaluate***: By implement this, results of models can be seen and even tabularize to improve visual quality.

ProFAB availabilty:
	Operating System: Platform independent (except Protein Feature Extraction which can be run in LINUX and MAC.)\
	Programming language: Python: >=3.7\
	Package Requirements: tqdm (4.63.0), requests 2.27.1), numpy (1.21.2), scikit-learn (1.0.1)

To get repository, execute following line:
```
git clone https://github.com/Sametle06/benchmark_platform.git
```

## How to implement modules:

The way to implement the functions in pythonic way is given in [test_file_1](use_case/test_file_1.ipynb) and [test_file_2](use_case/test_file_2.ipynb).

If one run the program in terminal, then exPro.py can be used. This method accepts multiple inputs however doesn't accept users datasets. Its parameters are:

- **file_name**: File includes dataset names such as GO_0000018, GO_1905523. If *isUser* = True or *isFasta* = True, then directory to dataset files must be defined in input file. Each must be defined in new line. 

- **score_path**: default = 'score_path.csv', A destination where scores are saved. It must be .csv file.
- **set_type**: {'random','similarity','temporal'}, default = 'random':
                split type of data, random:random splitting, target:
                similarity based splitting, temporal: splitting according to
                annotation time. If *isUser* or *isFasta* is True, random splitting
                will be applied to data.
- **protein_feature**: {'paac','aac','gaac','ctriad','ctdt','soc_number','kpssm'},
                default = 'paac': numerical features of protein sequences
- **ratio**: ratio: {None, float, list}, default = 0.2: used to split data 
                into train, test, validation sets as given values. If left None, 
                only X and y data can be obtained while float value gives train 
                and test set. If ratio = a (float), then test will be a% of total 
                data size. If ratio = [a,b] where a and b are in (0,1), 
                train, test and validation sets are formed according to them. For example, 
                If a = 0.2 and b = 0.1, train fraction is 0.7, test fraction is 0.2 
                and validation fraction is 0.1 of all dataset size. If set_type = 'temporal', 
                then ratio = None automatically.
- **pre_determined**: bool, default = False, if False, data is given
                according to ratio type, If True, already splitted data will
                provided.
- **scale_type**: {'normalizer','standard','max_abs','min_max','robust'}, default = 'standard, 
				determines the method to scale the data
- **ml_type**: {'logistic_reg','ridge_class','KNN','SVM','random_forest','MLP','naive_bayes', 
				decision_tree',gradient_boosting'}, default = "logistic_reg",
                Type of machine learning algorithm.

- **isFasta**:type = bool, default = False If True, a data provided by user is Fasta 
				file else numerical data should be introduced. While *isUser* = True, this parameter cannot be True
                at the same time. Format of fasta files must be **.fasta** and names of files should describe label.
                As an example, content of input file "sample_inputs.txt" should be like that:

                    directory_to_file/positive_data.fasta
                    directory_to_file/negative_data.fasta

- **place_protein_id**:type = int, default = 1, It indicates the place of protein id in fasta header.
               e.g. fasta header: >sp|O27002|....|....|...., seperate the header wrt.
               '|' then >sp is in the zeroth position, protein id in the first(1)
               position.

- **isUser**: type = bool, default = False, If True, user data path must be defined in file else ProFAB data
                will be used if data names are introduced correctly. While *isFasta* = True, this parameter cannot be True
                at the same time. If *label* = False, names of files should describe label. As an example, content of input
                file "sample_inputs.txt" should be like that:

                    directory_to_file/positive_data.txt
                    directory_to_file/negative_data.txt
    If **label** = True

                    directory_to_file/data.txt

- **delimiter**: type = str, default = "\t", a character to separate columns in file.
- **name**: type = bool, default = False, If True, then first colmun
            is considered as name of inputs else the first column is a 
            feature column.
- **label**: type = bool, default = False, If True, then last colmun
            is considered as label of inputs else the last column is a 
            feature column. 

It can be run in terminal with these lines:

where *isFasta* = False and *isUser* = False, use support vector as training algorithms and protein descriptor is CTRIAD and save perfomance
of model to *my_score_path.csv*:
```
python ezPro.py --file_name sample_inputs.txt --score_path my_score_path.csv --ml_type SVM --protein_feature ctriad
```
where *isFasta* = True, use k-nearest neighbor as training algorithm and ratio of test set over train set is 0.3:
```
python ezPro.py --file_name sample_inputs_userTrue.txt --ml_type KNN --ratio 0.3
```
where *isFasta* = False, use random forest as training algorithm:
```
python ezPro.py --file_name sample_inputs_fastaTrue.txt --ml_type random_forest
```


ProFAB does many jobs at the same time, therefore, reading the introductions and following the use cases given in sections are highly recommended. Detailed explanations can be found in each module: [import_dataset](profab/import_dataset), [model_preprocess](profab/model_preprocess), [model_learn](profab/model_learn), [model_evaluate](profab/model_evaluate).



## License

MIT License

ProFab Copyright (C) 2020 CanSyL

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
