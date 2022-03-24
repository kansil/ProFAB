# Import Dataset

## Importing from our database
In ProFAB, we tried to provide the datasets to user in almost every way. By defining different combinations of parameters, different outputs can be obtained.

### Explanation of parameters

- ***set_type***: {'random','similarity', 'temporal'}, (default = 'random'), is used to select the dataset spread. 'random' means data spread is random while 'similarity' means data points are separated according to their similarity then data is splitted randomly in importing. 'temporal' splitting is based on annotation time of inputs and no extra splitting is not done.
- ***ratio***: {None,float,list}, (default = 0.2), is used to split the data according given value(s). If left None, only X and y data can be obtained while float value gives train and test set. If ratio = [a,b] where a and b are in (0,1), train, test and validation sets can be obtained. If a = 0.2 and b = 0.1, train fraction is 0.7, test fraction is 0.2 and validation fraction is 0.1 of all dataset size. If set_type = 'temporal', then ratio = None automatically.
- **protein_feature**: {'paac','aac','gaac', 'ctdt','socnumber', 'ctriad', 'kpssm'}, (default = 'paac'), indicates numerical feature of proteins obtained from sequence data.
- ***label***: {None,'positive','negative'}, (default = None), user can get positive or negative sets of whole dataset by defining this parameter. If not None, only feature matrix will be returned.
- ***pre_determined***: {False,True}, (default = False), indicate how data will be get. We upload our dataset as train and test set. So user can get them without randomly foming the test and train sets from the whole data. 

### How to use

To import data, two functions ECNO (datasets for EC number) and GOID (datasets for GO term) can be used.

A use case for ECNO function use:
```{python}
from profab.import_dataset import ECNO
data_model = ECNO(ratio = [0.1, 0.2], protein_feature = 'paac', pre_determined = False, set_type = 'random')
X_train,X_test,X_validation,y_train,y_test,y_validation = data_model.get_data(data_name = 'ecNo_1-2-4')
```

A use case for GOID function use:
```{python}
from profab.import_dataset import GOID
data_model = GOID(ratio = [0.1, 0.2], protein_feature = 'paac', pre_determined = False, set_type = 'random')
X_train,X_test,X_validation,y_train,y_test,y_validation = data_model.get_data(data_name = 'GO_0000018')
```

## Importing self datasets

ProFAB allows users to implement their datasets to train in ProFAB learning modules thanks to function SelfGet(). 

### Explanation of parameters:

- ***delimiter***: default = "\t", a character to separate columns in file.
- ***name***: type = bool, default = False, If True, then first colmun
    is considered as name of inputs else the first column is a 
    feature column.
- ***label***: type = bool, default = False, If True, then last colmun
    is considered as label of inputs else the last column is a 
    feature column. 

### How to use

After importing, with a single line data can be imported as a python list:
```{python}
from profab.import_dataset import SelfGet
data = SelfGet(delimiter, name, label).get_data(file_name)
```

A use case:
```{python}
from profab.import_dataset import SelfGet
data = SelfGet(delimiter = '\t', name = False, label = False).get_data(file_name = "sample.txt")
```

After importing datasets, preprocessing and training can be done by following introductions in [model_preprocess](../model_preprocess) nad [model_learn](../model_learn).
