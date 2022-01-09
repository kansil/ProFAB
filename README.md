# ProFAB: Protein Functional Annotation Benchmark

This platform is generated to provide some machine learning algorithms like SVM, random forest etc. for the tasked dataset drug-target interaction, EC number prediction and GO ID prediction. This platform is based on supervised learning. 
Repository can be obtained by a single line of command:
```
git clone https://github.com/Sametle06/benchmark_platform.git
```
Operating System: Platform independent\
Programming language: Python: 3.7\
Package Requirements: numpy, scikit-learn, scipy

## Folders Description

This project is made to provide platforms to ease use of biological datasets, preprocessing steps and machine learning algorithms. The datasets are based on two concepts Enzyme Commission Number prediction(ecNoPred) and Gene Ontology Term prediction (goPred). These files include a lot of datasets from smaller size (n < 500) to larger size (n > 100000) to train with a numerous learning algorithms which can give optimized and tuned results by defining the name of function. Also, metrics based on recall, precision and accuracy are provided to see performances in many perspectives.

## How to run the machine learning algorithms 

The way to implement the functions of ProFAB is given in [test_file](test_file.ipynb).

If the one run the program for multiple datasets, then exPro.py can be used. Its parameters are:
- **file_name**: File includes dataset names such as GO_0000018, GO_1905523. Each name must be defined in new line.
- **score_path**: default = 'score_path.csv', A destination where scores are saved. It must be .csv file.
- **set_type**: {'random','similarity','temporal'}, default = 'random', Splitting type of train and test sets.
- **protein_feature**: {'aac','paac','gaac','ctriad','ctdt','socnumber','kpssm'}, default = 'paac', Numerical feature of protein sequence.
- **ratio**: type: {list, float, None}, default = 0.2, Ratio of between validation and test sets to train set.
- **pre_determined**: type: bool, default = False, If True, data will be given splitted train test sets else splliting will be done.
- **scale_type**: default = 'standard', Scaling of data to prevent biases.
- **ml_type**: default = 'logistic_reg', Machine learning algorithms will be used in prediction.

It can be run in terminal with this line:
```
python ezPro.py --file_name sample_inputs.txt --score_path my_score_path.csv
```

The parameters used in dataset importing are explained in [import_datasets](profab/import_dataset). Other steps are explained in [model_process](probab/model_process), [model_evaluate](profab/model_evaluate), and [feature_extraction](profab/feature_extraction_module).

## License

MIT License

ProFab Copyright (C) 2020 CanSyL

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
