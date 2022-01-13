# Import Dataset

## Importing from our database
In this project, we tried to present the datasets to user in almost every way. By defining different combinations of parameters, different outputs can be obtained.

The parameters:

- **set_type**: {'random','similarity', 'temporal'}, (default = 'random'), is used to select the dataset spread. 'random' means data spread is random while 'similarity' means data points are separated according to their similarity then data is splitted randomly in importing. 'temporal' splitting is based on annotation time of inputs and no extra splitting is not done.
- **ratio**: {None,float,list}, (default = 0.2), is used to split the data according given value(s). If left None, only X and y data can be obtained while float value gives train and test set. If ratio = [a,b] where a and b are in (0,1), train, test and validation sets can be obtained. If a = 0.2 and b = 0.1, train fraction is 0.7, test fraction is 0.2 and validation fraction is 0.1 of all dataset size. 
- **protein_feature**: {'paac','aac','gaac', 'ctdt','socnumber', 'ctriad', 'kpssm'}, (default = 'paac'), indicates numerical feature of proteins obtained from sequence data.
- **label**: {None,'positive','negative'}, (default = None), user can get positive or negative sets of whole dataset by defining this parameter. If not None, only feature matrix will be returned.
- **pre_determined**: {False,True}, (default = False), indicate how data will be get. We upload our dataset as train and test set. So user can get them without randomly foming the test and train sets from the whole data. 

## Importing self datasets

ProFAB allows users to implement their datasets to train in ProFAB learning modules thanks to function SelfGet(). 

The parameters:

- **delimiter**: default = "\t", a character to separate columns in file.
- **name**: type = bool, default = False, If True, then first colmun
    is considered as name of inputs else the first column is a 
    feature column.
- **label**: type = bool, default = False, If True, then last colmun
    is considered as name of inputs else the last column is a 
    feature column. 

After importing datasets, preprocessing and training can be done by following introductions in [model_preprocess](../model_preprocess) nad [model_learn](../model_learn).
