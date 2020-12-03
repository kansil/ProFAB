## Description of study, folders and files in the Repository

This study aims to classify the enzymes according to their Enzyme Comission numbers(ECn). To achieve that binary classification is used via Python 3.7 and corresponding packages e.g. scikit-learn. The labels are {-1,1} and feature matrix is based on protein amino acid sequences. 

### To do list and Folder Description

The main input file which consists of protein names and their corresponding ECn, annotation scores and amino acid sequences is downloaded from UniProt SwissProt. Then to increase the prediction quality, traning model is formed in different levels of EC numbers so that propagation of data was made in 4 level. For example, 1.-.-.- is level 1, 1.1.-.- is level 2, 1.1.1.- is level 3 and 1.1.1.1 is level 4. After the code made its job, 4 folders (level_1,level_2,level_3 and level_4)will form in ecn_propagation folder. Then UniProt-Uniref50 proteins are used to apply similarity based splitting function, too. The reason to do that is reducing the similarities between proteins and making model more generalizable. Therefore splitting function does two processes. The first one is random splitting and the seconde splitting function is similarity based.
After spliting is done, learning algorithms run and training starts. The outcome model is saved for future prediction and evaluation. 

The files are explained in the following lines:

- ***uniprot-reviewed_yes.tab*** : is the main input file that contains raw data. Columns it has are UniProt ID of proteins, entry name, gene names, Length, Enzyme Commission number, annotation scores and proteins amino acid sequences. As an example a lines is given:
```
UniProt_ID Entry name Gene names Length EC Number Annotation Sequence
C5VZW3  ACCD_STRSE  accD SSU1598    288 2.1.3.15    3 out of 5  MALFRKKDKYIRINPNRSRIESAPQAKPEVPDELFSKCPACKVILYKNDLGLEKTCQHCSYNFRITAQERRALTVDEGSFEELFTGIETTNPLDFPNYLEKLAATRQKTGLDEAVLTGKATIGGQPVALGIMDSHFIMASMGTVVGEKITRLFELAIEERLPVVLFTASGGARMQEGIMSLMQMAKISAAVKRHSNAGLFYLTVLTDPTTGGVTASFAMEGDIILAEPQTLVGFAGRRVIESTVRENLPDDFQKAEFLQEHGFVDAIVKRQDLPATISRLLRMHGGVR   
``` 
Our data, UniProtKB 2020 05 results, was downloaded on 06/11/2020, 
Due to its size, a supply link is provided: https://www.uniprot.org/uniprot/?query=reviewed:yes 
- ***target_featurization***: is a folder that is used to create the feature matrices for both train and validation datasets that are used in machine learning algorithms. 
	- **ec_propagation.py** : It is used to propagate the enzyme data that downloaded from UniProt SwissProt. 
	- **ec_split_methods.py** : splits the data into train and validation set and save them into the EC_level_L where L stands for level number.
	- **ec_fasta_to_array.py** : converts protein sequences .txt file to fasta format to use in ILearn tool
The order to follow:
```
-ec_propagation.py
-split_methods.py
-ec_fasta_to_array.py
-obtain the protein numerical features from iLearn tool: https://ilearn.erc.monash.edu/
```

- Data Processing and Final form is given in following figure:

![Map of Preparing Dataset ](https://user-images.githubusercontent.com/37181660/100945269-f10fe900-3511-11eb-8b9b-cb01fcf87b2a.PNG)

Finally, training and showing evaluation in a table were done in the following folder

- ***bin***: is a folder that involves main functions to run the machine learning algorithms
	- **ec_loading_features_indices.py** : is used to prepare the dataset to the main function
	- **ec_to_table** : to make a .csv file that shows the evaluation scores of model
	- **ec_main_setup** : main function to do job. The parameters it has:
- ***EC_level_1*** : is a folder where all processed data, training models prediction part and scores of validation set are stored. 


