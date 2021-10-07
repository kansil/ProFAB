# Import Dataset

## Importing from our database
In this project, we tried to present the datasets to the user in almost every way. User can get the set in splitted or directly or s/he can get only negative or positive datasets by defining the parameters clearly. 

The parameters shows difference for datasets are used in regression algoritm and sets in binary classification. Because of that, this section should be examined carefully to import the datasets.

The parameters:

- **set_type**: {'random','target'}, (default = 'random'), is used to select the dataset spread. 'random' means data spread is random while 'target' means data points are spreaded according to their similarity in clusters. Only cluster centers are collected as data. To get this data UniProt/UniRef50 dataset was used.
- **ratio**: {None,float,list}, (default = 0.2), is used to split the data according given value(s). If left None, only X and y data can be obtained while float value gives train and test set. If ratio = [a,b] where a and b are in (0,1), train, test and validation sets can be obtained. If a = 0.2 and b = 0.1, train fraction is 0.7, test fraction is 0.2 and validation fraction is 0.1 of all dataset size. 
- **protein_feature**: {'paac','aac','gaac', 'ctdc'}, (default = 'paac'), indicates numerical feature of proteins obtained from sequence data.
- **label**: {None,'positive','negative'}, (default = None), to obtain which set of data will be obtained. The user can get positive or negative sets of whole dataset by defining this parameter.
- **pre_determined**: {False,True}, (default = False), indicate how data will be get. We upload our dataset as train and test set. So user can get them without randomly foming the test and train sets from the whole data. 
