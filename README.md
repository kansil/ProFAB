# ProFAB: Protein Functional Annotation Benchmark

This platform is generated to provide some machine learning algorithms like SVM, random forest etc. for the tasked dataset drug-target interaction, EC number prediction and GO ID prediction. This platform is based on supervised learning. 
Repository can be obtained by a single line of command:
```
git clone https://github.com/Sametle06/benchmark_platform.git
```
Operating System: Platform independent  Programming language: Python: 3.7  Package Requirements: numpy, scikit-learn, scipy

## Folders Description

This project is made to provide some platforms that includes some pre-processed datasets(without scaling) and training algorithms. The platforms are based on Drug-Target Interaction prediction (dtiPred), Enzyme Commission Number prediction(ecNoPred) and Gene Ontology ID prediction (goPred). These files include a lot of datasets from small size (n < 500) to big size (n > 200000) to train with a numerous learning algorithms. Also, to train these datasets, we provide easy to use machine learning algorithm. By defining the name of function, optimized and tuned results can be obtained.

## How to run the machine learning algorithms 

The way to implement the code is given in [test_file](test_file.ipynb).

The parameters used in dataset importing are explained in [import_datasets](profab/import_dataset). Other steps are explained in [model_process](probab/model_process) and in [model_evaluate](profab/model_evaluate).

