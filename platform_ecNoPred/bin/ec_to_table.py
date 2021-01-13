import pandas as pd
import numpy as np


def to_table(score,size):
    
    columns = ['set','size','Precision','Recall','F1-Score','F05-Score','Accuracy','MCC']
    index = ['Train','Test','Validation']
    
    scores = []
    for j,i in enumerate(score):
        s = np.insert(i,0,index[j])
        s = np.insert(s,1,size[j])
        scores.append(s)
    df = pd.DataFrame(scores,columns = columns)
    return df

def form_table(score_path,scores,size):
  
    overall_table = to_table(np.array(scores,dtype = str),size).set_index(['set'])  
    with open(score_path,'w') as f:
        overall_table.to_csv(f,encoding='utf-8')

