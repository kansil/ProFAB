import numpy as np
import json
import re

from collections import defaultdict

def get_prot(file_name):
    target_list = np.loadtxt(file_name,dtype = 'str')
    return target_list #target_50

def get_fasta(fasta_file):
    fasta_data = defaultdict(list)    
    with open (fasta_file) as f:
        for i,rows in enumerate(f):
            rows = re.split(' |,',rows)
            for i in range(len(rows)-1):
                fasta_data[rows[i]].append(rows[-1])
    return fasta_data

def find_idx_fasta(folder, target,fasta_data):
    find_in_fasta = []
    with open ('../' + folder+'/fasta_data.fa','w') as sf:
        with open ('../' + folder + '/target_indices.txt','w') as f:
            for i in range(len(target)):
                
                try:
                    rows = fasta_data[target[i]][0]
                    lines = np.array([['>',target[i]]])

                    np.savetxt(sf,lines,fmt = '%s')
                    np.savetxt(sf,[rows[:-1]],fmt = '%s')
                    f.write(str(i))
                    f.write('\n')
                except:
                    pass
                # rows = np.where(fasta_data[:,0] == target[i])[0]

def do_all(folder,file_name,fasta_data):
    target = get_prot(file_name)
    find_idx_fasta(folder,target,fasta_data)

folder = 'kinase_data'
file_name = '../proj_data/' + folder + '/targets_ids_list.txt'
fasta_file = 'protein_fasta.txt'

fasta_data = get_fasta(fasta_file)
do_all(folder,file_name,fasta_data)




