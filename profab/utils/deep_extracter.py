# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 11:25:51 2022

@author: Sameitos
"""
import re
import numpy as np
import torch

def change_seq(seq_data,max_len):
    
    data = []
    for seq in seq_data:
        seq = re.sub(r"[UZOB]", "X", seq)
        j = ''
        for k,letter in enumerate(seq):
            if k == max_len:
                break
            j+=letter+' '
        data.append(j)
    return data

def t5_features(seq_data,take_avg,max_len):
    
    '''
    Description:
        This function is to transform protein sequences into continuous data
        by using RostLab pretrained model 'prot_t5_xl_uniref50' with "transformers"
        Python package
    Parameters:
        seq_data: {np.array, list}, protein sequence data. 
        take_avg: {bool}, default = False, if True, average of vectors will be returned
        max_len: {int}, default = -1, Max sequence lenght to embed
    Return:
        features: {np.array}, transformed continous data.
        
    '''
    
    from transformers import T5Tokenizer, T5Model
    
    seq_data = change_seq(seq_data, max_len)
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)
    
    model = T5Model.from_pretrained('Rostlab/prot_t5_xl_uniref50')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    
    
    embedded_data = []
    #for seq in seq_data:
    
    ids = tokenizer.batch_encode_plus(seq_data, add_special_tokens=True, padding=True)

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids = input_ids)[0]
            
    features = [] 
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        if take_avg:
            seq_emd = embedding[seq_num][1:seq_len-1].mean(dim=0).cpu().numpy()
        else:
            seq_emd = embedding[seq_num][1:seq_len-1]
        features.append(seq_emd)
    
    return features

def bert_features(seq_data,take_avg,max_len):
    
    
    '''
    Description:
        This function is to transform protein sequences into continuous data
        by using RostLab pretrained model "prot_bert" with "transformers" Python package
    Parameters:
        seq_data: {np.array, list}, protein sequence data. 
        take_avg: {bool}, default = False, if True, average of vectors will be returned
        max_len: {int}, default = -1, Max sequence lenght to embed
    Return:
        features: {list}, transformed continous data.
        
    '''
    

    seq_data = change_seq(seq_data,max_len)
    
    from transformers import BertModel, BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
    model = BertModel.from_pretrained('Rostlab/prot_bert')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device,'\n')
    
    model = model.to(device)
    model = model.eval()
    
    #embedded_data = []
    #seq_data = [re.sub(r"[UZOB]", "X", sequence) for sequence in seq_data]
    
    #for seq in seq_data:
        
    ids = tokenizer.batch_encode_plus(seq_data, add_special_tokens=True, padding=True)

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
    
    #print(embedding.size())

    features = [] 
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        if take_avg:
            seq_emd = embedding[seq_num][1:seq_len-1].mean(dim=0).cpu().numpy()
        else:
            seq_emd = embedding[seq_num][1:seq_len-1]
        features.append(seq_emd)
    
    return features



