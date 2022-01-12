## Description

ProFAB preprocessing step includes three modules featurization, splitting and scaling. All three can be used by users separately and without any limitations. 

### Featurization Module

Feature extraction module is a python-based tool that can be used to extract protein featured 
from a fasta file as an input.
There are 21 protein descriptors from POSSUM tool and 18 from iFeature tool.
The available protein descriptors and their abbreviations are indicated in the tables above.

Please, follow the steps below to run feature extraction module in the terminal (Linux/MAC).
The following code assumes that you work in the main directory of ProFAB.
The detailed explantion is in [feature_extraction](../utils/feature_extraction_module)

#### Usage

```{python}
from profab.model_preprocess import extract_protein_feature
feature_extracter.extract_protein_feature(protein_feature,
                                          place_protein_id,
                                          input_folder, 
                                          fasta_file_name)
```

A use case:
```{python}
from profab.model_preprocess import extract_protein_feature
feature_extracter.extract_protein_feature('edp', 1, 
                                          'profab/feature_extraction_module/input_folder', 
                                          'sample')
```

### Splitting Module

Splitting module is a python based tool that splits datasets into train, test and validation randomly. By defining fraction(s) of sets, it can be used.

#### Explanation of Parameters 

-**X**: Feature matrix that holds information of data
-**y**: Label matrix 
- **ratio**: type = {float,list}, (default = 0.2), is used to split the data according given value(s). If ratio = a (float), then test will be a% of total data size. If ratio = [a,b] where a and b are in (0,1), train, test and validation sets are formed according to them. If a = 0.2 and b = 0.1, train fraction is 0.7, test fraction is 0.2 and validation fraction is 0.1 of all dataset size. 

#### Usage

```{python}
from profab.model_preprocess import ttv_split
X_train,X_test,X_validation,y_train,y_test,y_validation = ttv_split(X, y, ratio)
```

A use case:
```{python}
from profab.model_preprocess import ttv_split
X_train,X_test,X_validation,y_train,y_test,y_validation = ttv_split(X, y, ratio = [0.1,0.2])
```

### Scaler Module

This module is to scale the data to new ranges to eleminate the biases and weigth differences between input points. The functions used are obtained from scikit-learn package of Python. The used functions are:
    - Normalizer
    - Standard_Scaler
    - maxAbs_Scaler
    - minMax_Scaler
    - Robust_Scaler


#### Explanation of Parameters

-**X_train**: type = {list, numpy array}, A data to train scaling functions
-**scale_type**: {'normalizer','standard','max_abs','min_max','robust'}, default = 'standard, determines the method to scale the data.

#### Usage

```{python}
from profab.model_preprocess import scale_methods
X_train,scaler = scale_methods(X_train,scale_type)
```

A use case:
```{python}
from profab.model_preprocess import scale_methods
X_train,scaler = scale_methods(X_train,scale_type = "standard")
X_test = scaler.transform(X_test)
```
