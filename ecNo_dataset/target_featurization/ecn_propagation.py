# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 21:10:03 2020

@author: Sameitos
"""

import os
import re 
import numpy as np

file_name = 'uniprot-reviewed_yes.tab'

folder = '../ecNo_propagated_data'

if not os.path.exists(folder):
    os.makedirs(folder)
    
with open(file_name) as f:
    for k,rows in enumerate(f):
        if k == 0:continue 
        rows = re.split('\t',rows.strip('\n'))
        if len(rows[-3]) <= 7:
            j = [rows[0], rows[-3], rows[-2][0],rows[-1]]
            a = [rows[0], rows[-2][0],rows[-1]]
            
            if j[1] !='':
                i = j[1].split('.')
                
                for level in range(1,5):
                    if not os.path.exists(folder + '/level_' + str(level)):
                        os.makedirs(folder + '/level_' + str(level))
                        
                with open(folder + '/level_1/ecNo_' + str(i[0]) + '.txt','a') as f:
                    np.savetxt(f,[j],fmt = '%s')
                
                if i[1] != '-':
                    with open(folder + '/level_2/ecNo_' + str(i[0]) + '-' + str(i[1])+ '.txt','a') as f:
                        np.savetxt(f,[j],fmt = '%s')
                        
                if i[2] != '-':
                    with open(folder + '/level_3/ecNo_' + str(i[0]) +'-' + str(i[1])  + '-' + str(i[2])+ '.txt','a') as f:
                        np.savetxt(f,[j],fmt = '%s')
                        
                if i[3] != '-':
                    with open(folder + '/level_4/ecNo_' + str(i[0]) + '-' + str(i[1]) + '-' + str(i[2]) + '-' + str(i[3]) + '.txt','a') as f:
                        np.savetxt(f,[j],fmt = '%s')
                
            if j[1] == '':
                with open(folder + '/no_ecNo' + '.txt','a') as f:
                    np.savetxt(f,[a],fmt = '%s')         

        # if k == 2000: break
        if k % 8000 == 0:print('checkpoint {}'.format(k))
            

        
        
        
        
        