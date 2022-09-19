# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 11:25:51 2022

@author: Sameitos
"""
import os, re
import numpy as np
import torch

def read_fasta_to_dict(input_dir, fasta_file, place_protein_id):
    """read_fasta_to_dict function is to read protein ids and sequences from the given fasta file.

    This funtions forms a dictionary of the fasta file. The keys of the dictionary are protein ids
    and values are the corresponding protein sequences

    Parameters:
        input_dir (str): it is full path to the directory that contains fasta file
        fasta_file (str): it is the name of the fasta file without fasta extension
        place_protein_id (int): it is to define where the protein id places in the fasta header
        when it is splitted according to | sign.

    Returns:
        dict: This is a dict of fasta file. The keys of fasta_dict are protein ids and
        values are protein sequences.

    """
    fasta_dict = set()#dict()
    seq_list = []
    sequence = ""
    prot_id = ""
    with open("{}/{}.fasta".format(input_dir, fasta_file), "r") as fp:
        for line in fp:
            if line[0] == '>':
                if prot_id != "" and prot_id not in fasta_dict:
                    fasta_dict.add(prot_id)#[prot_id] = sequence
                    seq_list.append(sequence)
                prot_id = line.strip().split("|")[place_protein_id]
                if place_protein_id == 0:
                    prot_id = prot_id[1:]
                if prot_id not in fasta_dict:
                    
                    sequence = ""
            else:
                sequence += line.strip()
        seq_list.append(sequence)
        fasta_dict.add(prot_id)#[prot_id] = sequence
    fp.close()
    return seq_list

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

def t5_features(fasta_file, input_dir, place_protein_id,take_avg,max_len,output_folder):
    
    '''
    Description:
        This function is to transform protein sequences into continuous data
        by using RostLab pretrained model 'prot_t5_xl_uniref50' with "transformers"
        Python package
    Parameters:
        fasta_file: {str}, fasta file that includes protein sequence data. 
        take_avg: {bool}, default = False, If False, output will be saved as torch.tensor
                          if True, average of vectors will be saved as array. 
        max_len: {int}, default = -1, Max sequence lenght to embed
        input_dir (str): it is full path to the directory that contains fasta file
        fasta_file (str): it is the name of the fasta file without fasta extension
        place_protein_id (int): it is to define where the protein id places in the fasta header
        when it is splitted according to | sign.
    Return:
        features: {np.array}, transformed continous data.
        
    '''
    
    if output_folder is None:
        output_file = fasta_file + '_t5_xl.txt'
    else:
        output_file = output_folder + '/' + fasta_file + '_t5xl.txt'
    
    seq_data = read_fasta_to_dict(input_dir, fasta_file, place_protein_id)
    seq_data = change_seq(seq_data, max_len)    
    
    from transformers import T5Tokenizer, T5Model
    
    
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
            features.append(seq_emd)
        else:
            if not os.path.exists(output_file[:-4]+'/'):
                os.path.makedirs(output_file[:-4]+'/')
            torch.save(embedding[seq_num][1:seq_len-1],output_file[:-4]+'/feat_'+seq_num+'.txt')
    
    if features:
        np.savetxt(output_file,features)

    if not take_avg: return output_file[:-4]+'/'
    return output_file

    

def bert_features(fasta_file, input_dir, place_protein_id,take_avg,max_len,output_folder):
    
    '''
    Description:
        This function is to transform protein sequences into continuous data
        by using RostLab pretrained model 'prot_t5_xl_uniref50' with "transformers"
        Python package
    Parameters:
        fasta_file: {str}, fasta file that includes protein sequence data. 
        take_avg: {bool}, default = False, If False, output will be saved as torch.tensor
                          if True, average of vectors will be saved as array. 
        max_len: {int}, default = -1, Max sequence lenght to embed
        input_dir (str): it is full path to the directory that contains fasta file
        fasta_file (str): it is the name of the fasta file without fasta extension
        place_protein_id (int): it is to define where the protein id places in the fasta header
        when it is splitted according to | sign.
    Return:
        features: {np.array}, transformed continous data. 
    '''
    
    if output_folder is None:
        output_file = fasta_file + '_bert.txt'
    else:
        output_file = output_folder + '/' + fasta_file + '_bert.txt'
    seq_data = read_fasta_to_dict(input_dir, fasta_file, place_protein_id)
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
            features.append(seq_emd)
        else:
            if not os.path.exists(output_file[:-4]+'/'):
                os.path.makedirs(output_file[:-4]+'/')
            torch.save(embedding[seq_num][1:seq_len-1],output_file[:-4]+'/feat_'+seq_num+'.txt')
            
    if features:
        np.savetxt(output_file,features)

    if not take_avg: return output_file[:-4]+'/'
    return output_file



