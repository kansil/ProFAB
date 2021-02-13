## Description of folders and files in the Repository
- ***proj_data***: is folder that includes compound name, target name and bioactivity results between them. The folder inside is **nr_data** and it includes 3 file these are:
    - **bioactivity_data.csv**: is a file that keeps bioactivity results btw compound and target.(unit: nM) It is a sparse matrix such as:
    [['nan','nan',...],['nan',7.21354684,'nan'],...]
    - **compound_ids_list.txt**: is a file that contains Chembl IDs of compound. It look like : [CHEMBL3645270,CHEMBL3314028,CHEMBL3957820,...].transpose()
    - **target_ids_list.txt**: is a file that contains Chembl IDs of targets. It looks like: [CHEMBL4797,CHEMBL2269,CHEMBL5304,...].transpose()
**Note**: *Other folders and files will occur as directives dedicated in compound_filtration and target_filtration folders are followed*
- ***nr_data***: is a folder where all feature matrices, models and metrics are stored. It will be seen and filled with the files as following programs are executed. 
- ***compound_filtration***: This folder is used to define the compounds in their Morgan Fingerprints by using their SMILES demonstrations. The files are in the folder:
    - **compound_get_data.txt**: supplies an adress to download the relevant files to form the dataset
    - **get_smiles.py**: gets SMILES data for compound(CHEMBL_ID) and write on a file to use later.
    - **load_cluster.py**: clusters the compound according to their Morgan fingerprints then save the indices of the compounds
    - **whole.py**: aranging the file of chembl27_chemreps.txt to use in get_smiles.py
    - **compound_setup.py**: setup py file for compound features. 
    - **compound_main.py**: main function compounds. Parameters it has:
    	- **dataset_name**: is a name of folder where files are stored 
    	- **cluster**:bolean,(default:False): It decides to run clustering algorithm on data. If cluster = False, no cluster and whole compound data will be used in the training section otherwise only cluster senters will be used.
```
- The order will be followed is:
    - dowload the files from dedicated links in get_data.txt
    - execute whole.py
    - python compound_main.py --dataset_name: nr_data --cluster: False
```
The files will occur after running these programs in 'nr_data' folder are:
```
Compound_Smiles_Idx.txt: in-between file
representatives.json: cluster centers of compounds, (optional, if cluster:True)
compound_features.txt: main file will be used in traning 
```
- ***target_filtration***: This folder includes the files that define targets for dedicated dataset.The files are in the folder:
    - **get_data.txt**: is a file that provides a link where UnitProt Uniref %50 similarity data can be dowloaded
    - **chembl_27.fa**: a fasta file of targets obtained by Chembl
    - **protein_fasta.txt**: includes fasta features of targets convert from chembl_27.fa
    - **chembl_uniprot_mapping.txt**: is a file shows Chembl ID of targets and their corresponding UnitProt IDs
    - **READ_uniref.txt**: includes a link where needed files can be downloaded.
    - **get_target_feature.py**: write the pre_determined target features and their original indices to a file 'dataset_name/target_features.txt'.
    - **protein_get_data.py**: get the representatives from the clusters determined in the file 'uniref-identity.tab' and write them to a file 'uniref_reps.txt'. 
    - **protein_main.py**: is the main module that run the the file 'protein_selection.py'. The parameters:
        - **dataset_name**: data name where final data are saved
        - **similarity**: Target similarity in percent. For example, for targets show 30% similarity, it should be 30
    - **protein_selection.py**: includes the functions that find the target names written on 'target_ids_list.txt' file in the file 'uniref_reps.txt' and write these targets to a file 'dataset_name/cluster_protein' + 'similarity' + '_idx.json' where similarity indicates target similarities in uniref-identity.
    - **fasta_to_array.py**: obtains the fasta feature of dedicated targets in proj_data/nr_data/target_ids_list.txt and save them in a folder nr_data.
```
- The order will be followed is:
    - dowload the files from dedicated links in READ_uniref.txt
    - fasta_to_array.py
    - protein_get_data.py,(optional, if target based splitting is needed) 
    - protein_main.py,(optional, if target based splitting is needed)
    -'obtain PAAC feature of targets from IFEATURE toolkit ' 
    - get_target_feature.py
```
The files will occur after following the order in 'nr_data' folder are:
```
fasta_data.txt: Fasta properties of target
PAAC.tsv: pseudo amino acid composition feature matrix of targets
cluster_protein50_idx.json (optional, if protein_get_data.py and protein_main.py files were executed)
target_features.txt: main file will be used in traning
```
After preparing both compound and target feature matrices, they are joined to each other in axis=1 (column wise) and label matrix where bioactivity result are recorded added this feature matrix. For easy understanding, following figure is provided:

![prefinal](https://user-images.githubusercontent.com/37181660/95652174-f1829980-0af7-11eb-94ce-cc7b4d51b959.PNG)

Pre-Final Dataset Form is the format that training algorithms will accept. So above files should be run at once if your dataset is not in the same format as indicated in the figure. If the data will be introduced to the machine has the same format as Pre-Final Dataset Form, above sections can be passed. However in your data folder ('nr_data' for our case) following files must exist:
```
- compound_features.txt
- target_features.txt
- bioactivity_data.csv
```
If similarity based splitting methods will be applied, then following files should exists too:
```
- representatives.txt(for compound based splitting)
- cluster_protein50_idx.txt(50 shows target similarities by 50%. For 30% similarity it will be 30)(for target based splitting)
- For compound & target based splitting, both must exist
```
Source code folder contains.

- ***bin***: is folders that keeps source codes inside. These are:
    - **main_setup.py**: It is the main() function that applies the following modules. 
    - **loading_feature_indices.py**: It changes dataset its pre-final form to its final form as shown in following figure:![final](https://user-images.githubusercontent.com/37181660/95652292-acab3280-0af8-11eb-89b0-8e6e75989ea0.PNG)
    - **splitting_means.py**: It is the module to apply different splittin algorithms on dataset.These algorithms are: 
 		- *Random Splitting*: no info is used to split the data
 	    - *Similarity based* Splitting: similartiy matrix of data is used to split. Splitting is done by three means:
   		    - *Only Compound based*: compound are clustered by their FingerPrints and their centers are used to train model
   			- *Only Target based*: get the centers of clustered data of target by UnitProt Uniref(%50 similarity) and their centers are used to train model
   			- *Compound & Target based*: both compounds and targets are clustered by their similarities and their centers of clusters are used to train the model
