import pandas as pd
import numpy as np


def to_table(score,size):
    
    columns = ['set','size','mse','rmse','rse_score','pearson']
    index = ['Train','Test','Validation']
    
    scores = []
    for j,i in enumerate(score):
        s = np.insert(i[:-1],0,index[j])
        s = np.insert(s,1,size[j])
        scores.append(s)
    print('\nScore Table:\n',scores)
    df = pd.DataFrame(scores,columns = columns)
    return df

def form_table(score_path,scores,size):
  
    overall_table = to_table(np.array(scores,dtype = str),size).set_index(['set'])  
    with open(score_path,'w') as f:
        overall_table.to_csv(f,encoding='utf-8')

