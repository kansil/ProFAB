import pandas as pd
import numpy as np


def to_table(split_type,score,size):
    
    columns = ['split_type','set','size','mse','rmse','rse_score','pearson','average_AUC'] + list(score[0,5].keys())
    index = ['Train','Test','Validation']
    st = [split_type,split_type,split_type]
    scores = []
    for j,i in enumerate(score):
        s = np.insert(i[:-1],0,index[j])
        s = np.insert(s,0,st[j])
        s = np.insert(s,2,size[j])
        s = np.append(s,list(i[5].values()))
        scores.append(s)
    df = pd.DataFrame(scores,columns = columns)
    return df

def form_table(score_path,splitter_types,scores,size):
  
    overall_table = to_table(splitter_types,np.array(scores),size).set_index(['split_type','set'])  
    with open(score_path,'w') as f: 
        overall_table.to_csv(f,encoding='utf-8')

