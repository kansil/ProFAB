import numpy as np
import json
import re
import os 
from collections import defaultdict


folder = '../EC_level_1'
files = os.listdir(folder)
print(files)
for j,i in enumerate(files):
    for s in ['target_split','random_split']:
        files_folder = folder + '/' +  i + '/' + s
        for fs in os.listdir(files_folder):
            fs = files_folder + '/' + fs
            if fs[0] != 'f':
                with open (fs[:-3] + 'fa','w') as sf:
                    with open(fs) as f:
                        for rows in f:
                            rows = re.split(' ',rows.strip('\n'))
                            if len(rows[-1])>31:
                                lines = np.array([['>' + rows[0] + '|' + str(1) + '|' + 'training']])
                                np.savetxt(sf, lines, fmt = '%s')
                                rows = rows[-1]
                                np.savetxt(sf, [rows[:-1]], fmt = '%s')


folder = '../EC_level_2'
files = os.listdir(folder)
print(files)
for j,i in enumerate(files):
    for s in ['random_split']:
        files_folder = folder + '/' +  i + '/' + s
        for fs in os.listdir(files_folder):
            fs = files_folder + '/' + fs
            if fs[0] != 'f':
                with open (fs[:-3] + 'fa','w') as sf:
                    with open(fs) as f:
                        for rows in f:
                            rows = re.split(' ',rows.strip('\n'))
                            if len(rows[-1])>31:
                                lines = np.array([['>' + rows[0] + '|' + str(1) + '|' + 'training']])
                                np.savetxt(sf, lines, fmt = '%s')
                                rows = rows[-1]
                                np.savetxt(sf, [rows[:-1]], fmt = '%s')
