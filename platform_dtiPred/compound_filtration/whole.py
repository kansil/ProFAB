
import numpy as np
import re


"""
this function is used to extract compound names and their SMILES data 
from chembl_27_chemreps.txt file to the file 'whole.txt' 
"""


whole = np.array([]).reshape(0,2)
with open ('whole.txt', 'w') as sf:
    with open('chembl_27_chemreps.txt') as f:
        for i,rows in enumerate(f):
            rows =re.split('\t',rows.strip('n\r'))
            if i != 0: 
                np.savetxt(sf,np.array([np.append(rows[0],rows[1])]),fmt ='%s')
            
            