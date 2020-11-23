# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 19:01:23 2020

@author: Sameitos
"""

import numpy as np
import re 
from tqdm import tqdm
import os 
"""
This function get the representatives of clusters in uniref files (uniref50 & 90)
and stores them in different folders
"""

similarity = 50
folder = 'uniref' + str(similarity)
if not os.path.exists(folder):
    os.makedirs(folder)

s = open(folder + '/uniref' + str(similarity) + '_reps.txt','w')
with open(folder + '/uniref-filtered-identity_' + str(similarity/100) + '.tab') as f:
    for i, rows in enumerate(tqdm(f)):
        rows =re.split('\t',rows.strip('n\r'))
        if rows[1] != 'unreviewed and UniParc' and rows[1] != 'UniParc' and i!=0:
            rows_prot =re.split(';',rows[-1])[0].strip('\n\r')
            s.writelines(rows_prot+'\n')
s.close()
