# ProFAB: Protein Functional Annotation Benchmark

This platform is generated to provide datasets amd shallow machine learning algorithms such as SVM, random forest etc. for protein function prediction. Its learning method is supervised and to present complete set of machine learning study preprocessing and evaluation steps are added, too. ProFAB's main purpose is for protein based studies but its functions can be used separately for different reasons. To illustrate, one can use only to import and splitting data while another one can use only machine learning algorithms thank to its flexible structure.

![ProFAB Main Functions and Their Tasks](https://user-images.githubusercontent.com/37181660/149328976-fb7c81d0-9ba5-4ec3-a23e-1daa754c5e81.png)

As seen from the figure, in ProFAB, four main modules [import_dataset](profab/import_dataset), [model_preprocess](profab/model_preprocess), 
[model_learn](profab/model_learn), [model_evaluate](profab/model_evaluate) are employed.
- ***import_dataset***: This module is to load dataset from the files. Users can use ProFAB datasets or can load their data with related functions.
- ***model_preprocess***: To feed the data to learning algorithms, this module provides there pre-works:
	- featurization of protein sequence data
	- random splitting
	- scaling
- ***model_learn***: To apply machine learning algorithms
- ***model_evaluate***: By this, results of models can be seen and even tabularize to improve visual quality.

ProFAB availabilty:\
	Operating System: Platform independent (except Protein Feature Extraction which can be run in LINUX and MAC.)\
	Programming language: Python: >=3.7\
	Package Requirements: numpy, scikit-learn, scipy

To get repository, execute following line:
```
git clone https://github.com/Sametle06/benchmark_platform.git
```

## How to implement modules:

The way to implement the functions in pythonic way is given in [test_file](test_file.ipynb).

If one run the program in terminal, then exPro.py can be used. This method accepts multiple inputs however doesn't accept users datasets. Its parameters are:
- **file_name**: File includes dataset names such as GO_0000018, GO_1905523. Each name must be defined in new line.
- **score_path**: default = 'score_path.csv', A destination where scores are saved. It must be .csv file.
- **set_type**: {'random','similarity','temporal'}, default = 'random', Splitting type of train and test sets.
- **protein_feature**: {'aac','paac','gaac','ctriad','ctdt','socnumber','kpssm'}, default = 'paac', Numerical feature of protein sequence.
- **ratio**: type: {list, int, none}, default = 0.2, Ratio of between validation and test sets to train set.
- **pre_determined**: type: bool, default = False, If True, data will be given splitted train test sets else splliting will be done.
- **scale_type**: default = 'standard', Scaling of data to prevent biases.
- **ml_type**: default = 'logistic_reg', Machine learning algorithms will be used in prediction.

It can be run in terminal with this line:
```
python ezPro.py --file_name sample_inputs.txt --score_path my_score_path.csv
```

ProFAB does many jobs at the same time, therefore, reading the introductions and following the use cases given in sections are highly recommended. Detailed explanations can be found in each module: [import_dataset](profab/import_dataset), [model_preprocess](profab/model_preprocess), [model_learn](profab/model_learn), [model_evaluate](profab/model_evaluate).



## License

MIT License

ProFab Copyright (C) 2020 CanSyL

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
