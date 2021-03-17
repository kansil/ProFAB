# Compound Featuring via rdkit Python Package

Here, we tried to ease to obtain the numerical features of Chembl Compounds by using Python package 'rdkit'. By simply defining your file name you can get your categorical data of compounds. The results can strored as tab seperated, comma seperated and space separated.
In result file, except from the Chembl_IDs of molecules and their bitvectors. Also, a file that includes indices of the original data points is supplied. It is because rdkit package may give none value for some SMILES values. To not lose the places of points, indices of molecules were also supplied in the 2nd file. 

## Folders Description

- **compound_setup**: is a file that contain do run the other functions. 

- **get_smiles**: is a file to get SMILES data of compounds by Chembl_ID of molecules and change them to bitcectors.

- **chembl27_chemreps**: is a file that contains Chembl molecules SMILES and InChi data.

By writing the following code to the terminal, categorical values of molecules can be obtained:
```
python compound_to_ --data_name smiles.txt --save_data feature.txt --save_idx compound_indices.txt --bits 1024
```
## Output
```
	- feature.txt: contains Chembl Molecules and BitVectors by rdkit.
	- indices.txt: contains indices of original data.
```