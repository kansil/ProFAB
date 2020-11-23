import numpy as np
import json
import re
import os 
from collections import defaultdict


folder = '../EC_level_1'
files = os.listdir(folder)

for j,i in files:
    for s in ['similarty_based','random']:
            
        file_name = folder + '/class_' + str(j) + s + i
        if i[0] != 'f':
            with open (file_name[:-3] + 'fa','w') as sf:
                with open(file_name) as f:
                    for rows in f:
                        rows = re.split(' ',rows.strip('\n'))
                        if len(rows[-1])>31:
                            lines = np.array([['>' + rows[0] + '|' + str(1) + '|' + 'training']])
                            np.savetxt(sf, lines, fmt = '%s')
                            rows = rows[-1]
                            np.savetxt(sf, [rows[:-1]], fmt = '%s')


