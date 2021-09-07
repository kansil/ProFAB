# Import Dataset

## Importing from our database
In this project, we tried to present the datasets to the user in almost every way. User can get the set in splitted or directly or s/he can get only negative or positive datasets by defining the parameters clearly. 

The parameters shows difference for datasets are used in regression algoritm and sets in binary classification. Because of that, this section should be examined carefully to import the datasets.

The parameters:
- Regression based dataset importing:
	- **set_type**: {'random','target'}, (default = 'random'), is used to select the dataset spread. 'random' means data spread is random while 'target' means data points are spreaded according to their similarity in clusters. Only cluster centers are collected as data. To get this data UniProt/UniRef50 dataset was used.
	- **ratio**: {None,float,list}, (default = 0.2), is used to split the data according given value(s). If left None, only X and y data can be obtained while float value gives train and test set. If ratio = [a,b] where a and b are in (0,1), train, test and validation sets can be obtained. If a = 0.2 and b = 0.1, train fraction is 0.7, test fraction is 0.2 and validation fraction is 0.1 of all dataset size.  
	- **protein_feature**: {'paac','aac','gaac', 'ctdc'}, (default = 'paac'), indicates numerical feature of proteins obtained from sequence data.

- Classification based dataset importing:
	- **set_type**: {'random','target'}, (default = 'random'), is used to select the dataset spread. 'random' means data spread is random while 'target' means data points are spreaded according to their similarity in clusters. Only cluster centers are collected as data. To get this data UniProt/UniRef50 dataset was used.
	- **ratio**: {None,float,list}, (default = 0.2), is used to split the data according given value(s). If left None, only X and y data can be obtained while float value gives train and test set. If ratio = [a,b] where a and b are in (0,1), train, test and validation sets can be obtained. If a = 0.2 and b = 0.1, train fraction is 0.7, test fraction is 0.2 and validation fraction is 0.1 of all dataset size. 
	- **protein_feature**: {'paac','aac','gaac', 'ctdc'}, (default = 'paac'), indicates numerical feature of proteins obtained from sequence data.
	- **label**: {None,'positive','negative'}, (default = None), to obtain which set of data will be obtained. The user can get positive or negative sets of whole dataset by defining this parameter.
	- **pre_determined**: {False,True}, (default = False), indicate how data will be get. We upload our dataset as train and test set. So user can get them without randomly foming the test and train sets from the whole data. 

Importing from the database can be done via following codes:
![import_ec](https://user-images.githubusercontent.com/37181660/111300299-fd075d00-8661-11eb-9e73-a33c8a11bf2b.PNG)

## Importing User Datasets

### Dataset for Regression method:

To commit this import, user has to have feature matrix and its corresponding label matrix. These two matrix can be presented in different files and in different formats. The formats are '.tsv': tab_separated, '.txt': space separated or '.csv': comma separated. User can define in these three format their files. The files presented must be name that (example file format is .txt):

- X and y in one file:
	- **feature_label_dataset.txt**: Includes both features and their corresponding labels of data points.
- X and y in different files:
	- **feature_dataset.txt**: Includes only features of data points.
	- **label_dataset.txt**: Includes only label of data points.
- X and y in one file and dataset is already splitted:
	- **train_feature_label_dataset.txt**: Includes features and labels of train set
	- **test_feture_label_dataset.txt**: Includes features and labels of test set
- X and y in different file and dataset is already splitted:
	- **train_feature_dataset.txt**: Includes only features of train data points.
	- **train_label_dataset.txt**: Includes only label of train data points.
	- **test_feature_dataset.txt**: Includes only features of test data points.
	- **test_label_dataset.txt**: Includes only label of test data points.

If user have any type of sets, program will run automatically. Otherwise user will get the error: *FileNotFoundError*

### Datasets for Classification method:

To commit these import, user may have feature matrix which is splitted in positive and negative or user may have feature matrix with labels. These two matrix can be presented in different files and in different formats. The formats are '.tsv': tab_separated, '.txt': space separated or '.csv': comma separated. User can define in these three format their files. The files presented must be name that (example file format is .txt):

- X and y in one file:
	- **feature_label_dataset.txt**: Includes both features and their corresponding labels of data points.
- X and y in different files:
	- **feature_dataset.txt**: Includes only features of data points.
	- **label_dataset.txt**: Includes only label of data points.
- X and y in one file and dataset is already splitted:
	- **train_feature_label_dataset.txt**: Includes features and labels of train set
	- **test_feture_label_dataset.txt**: Includes features and labels of test set
- X and y in different file and dataset is already splitted:
	- **train_feature_dataset.txt**: Includes only features of train data points.
	- **train_label_dataset.txt**: Includes only label of train data points.
	- **test_feature_dataset.txt**: Includes only features of test data points.
	- **test_label_dataset.txt**: Includes only label of test data points.
- Positve and negative sets are separated:
	- **positive_dataset.txt**: Includes only positive data points
	- **negative_dataset.txt**: Includes only negative data points
- Positive and negative sets are separated and dataset is already splitted:
	- **positive_train_dataset.txt**: Includes only features of train data points.
	- **negative_train_dataset.txt**: Includes only label of train data points.
	- **positive_test_dataset.txt**: Includes only features of test data points.
	- **negative_test_dataset.txt**: Includes only label of test data points.

If user have any type of sets, program will run automatically. Otherwise user will get the error: *FileNotFoundError*