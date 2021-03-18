# Compound Featurization via rdkit Python Package

Here, we tried to ease to obtain the numerical features of Chembl Compounds by using Python package 'rdkit'. By simply defining your file name you can get your categorical data of compounds. The results can strored as tab seperated, comma seperated and space separated.
In result file, except from the Chembl_IDs of molecules and their bitvectors. Also, a file that includes indices of the original data points is supplied. It is because rdkit package may give none value for some SMILES values. To not lose the places of points, indices of molecules were also supplied in the 2nd file. 

## Folders Description

- **compound_setup**: is a file that contain do run the other functions. 

- **get_smiles**: is a file to get SMILES data of compounds by Chembl_ID of molecules and change them to bitcectors.

- **chembl27_chemreps**: is a file that contains Chembl molecules SMILES and InChi data.

## The Parameters:

- **data_name**: SMILES data of molecules
- **save_data**: (default: 'feature.txt'), name of file where fingerprits are stored. Its format can be .csv, .txt and .tsv
- **save_idx**: (default: indices.txt), name of file where indices of data points are stored. It is beacuse some data points can be lost during process.
- **bits**: (default: 1024), number of dimensions of fingerprints
