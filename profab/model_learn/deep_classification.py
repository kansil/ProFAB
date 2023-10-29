# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 17:00:59 2022

@author: Sameitos
"""
import os
import numpy as np
from sklearn.model_selection import RepeatedKFold, PredefinedSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import math
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def batch_loader(X,y,batch_size):
    
    train_data = []
    for i in range(len(X)):
        train_data.append([X[i],y[i]])
    
    return DataLoader(train_data,batch_size=batch_size)


class RNNet(nn.Module):
    
    def __init__(self,embedding_size,
                 out_size,
                 num_layers,
                 p,
                 ho,
                 cls_num=1):
        
        super(RNNet,self).__init__()
        
        self.cls_num = cls_num
        self.embedding_size = embedding_size
        # self.num_words = num_words
        # self.seq_len = seq_len
        
        self.out_size = out_size
        self.num_layers = num_layers
        
        self.ho = ho
        
        #self.embedding = nn.Embedding(self.num_words+1, self.embedding_size, padding_idx = 0)
        
        self.rnn = nn.RNN(self.embedding_size, self.out_size, num_layers = self.num_layers,
                          dropout = p)
        
        self.linear = nn.Linear(self.out_size, cls_num)

        
    
    def forward(self,x):
        
        #embeddings = self.embedding(x)
        #print(x.size())
        output, hidden = self.rnn(x, self.ho)
        
        out = self.linear(output[:,-1])
        if self.cls_num>1:
            out = F.softmax(out,dim = 1)
        return out


def rnn_classifier(X_train, y_train, X_valid, y_valid, rnn_params, model_path):
    
    
    embedding_size = rnn_params['embedding_size']
    epochs = rnn_params['epochs']
    lr = rnn_params['learning_rate']
    eps = rnn_params['eps']
    batch_size = rnn_params['batch_size']
    bidirectional = rnn_params['bidirectional']
    
    
    out_size = rnn_params['out_size']
    p = rnn_params['p']
    num_layers = rnn_params['num_layers']
    
    
    nfold = rnn_params['nfold']
    
    #X_train,encoder = onehot_encoder(X_train, y_train)
    #if X_valid:
    #    X_valid = encoder.transform(X_valid)
    if isinstance(X_train,(list,np.ndarray)):
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)
        if X_valid is not None:
            X_valid = torch.tensor(X_valid)
            y_valid = torch.tensor(y_valid)
    
    if len(X_train.size()) == 2:
        X_train = X_train.unsqueeze(1)
        n = 1
        if X_valid is not None:
            X_valid = X_valid.unsqueeze(1)
    else:
        n = len(X_train[0])
        
    
    if bidirectional:
        num_layers = 2*num_layers
    
    ho = torch.randn(num_layers, n, out_size).to(device)

    if y_train.unsqueeze(1).size()[-1]>1:
        model = RNNet(len(X_train[0][0]),out_size,
                     num_layers,p,ho,y_train.unsqueeze(1).size()[-1]).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        model = RNNet(len(X_train[0][0]),out_size,
                     num_layers,p,ho).to(device)
        criterion = nn.BCEWithLogitsLoss()
        y_train = y_train.unsqueeze(1)
        y_valid = y_valid.unsqueeze(1)
        
    if model_path is not None:

        if os.path.isfile(model_path):
            print('model already exists, it is loading..')
            model.load_state_dict(torch.load(model_path))
            return model
        
    optim = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=eps)
    
    
    
    if X_valid is not None:
        
        best_loss = float('inf')
        train_batch_loader = batch_loader(X_train,y_train,batch_size)
        
        for epoch in range(epochs):
            
            train_loss = 0.0
            valid_loss = 0.0
            
            model.train()
            for x_tr,y_tr in train_batch_loader:
                
                optim.zero_grad()
                pred = model(x_tr.to(device).float())
                #print(pred)
                loss = criterion(pred,y_tr.to(device).float())
                train_loss += float(loss.item())
                loss.backward()
                optim.step()
            
            
            model.eval()
            with torch.no_grad():
                
                pred_test = model(X_valid.to(device).float())
                loss = criterion(pred_test,y_valid.to(device).float())
                valid_loss+=loss.item()
                
            print(f'epoch: {epoch+1} valid loss: {valid_loss}, train_loss: {train_loss}')
            
            if best_loss>valid_loss:
                best_loss = valid_loss
                best_model = model
            if train_loss<best_loss:break
        if model_path is not None:
            print('saving model..')
            torch.save(best_model.state_dict(),model_path)
        return best_model
            
    else:
        prf = open('predY.txt','w')
        rkf = RepeatedKFold(n_splits = nfold, n_repeats = 5, random_state = 10000)
        best_loss = float('inf')
        for epoch in range(epochs):
            
            train_loss = 0.0
            valid_loss = 0.0
            
            
            
            for train_idx,valid_idx in rkf.split(X_train):
                
                train_batch_loader = batch_loader(X_train[train_idx],y_train[train_idx],batch_size)
                
                model.train()
                for x_tr,y_tr in train_batch_loader:
                    
                    optim.zero_grad()
                    
                    pred = model(x_tr.to(device).float())
                    #print('ohh noooo')
                    #for i in range(len(pred)):
                    #    prf.write(f'{pred[i]} {y_tr[i]}\n')
                    loss = criterion(pred,y_tr.to(device).float())
                    train_loss += float(loss.item())
                    loss.backward()
                    optim.step()
                
                
                model.eval()
                with torch.no_grad():
                    pred_test = model(X_train[valid_idx].to(device).float())
                    loss = criterion(pred_test,y_train[valid_idx].to(device).float())
                    valid_loss+=loss.item()
                

            if best_loss>valid_loss:
                best_loss = valid_loss
                best_model = model
            print(f'epoch: {epoch+1} valid loss: {valid_loss}, train_loss: {train_loss}')
            if train_loss< best_loss:break
        for i in range(len(pred)):
            prf.write(f'{pred[i]} {y_tr[i]}\n')
        prf.close()
        if model_path is not None:
            print('saving model..')
            torch.save(best_model.state_dict(),model_path)
        return best_model
    

class CNNet(nn.Module):
    
    
    def __init__(self,embedding_size,seq_len,
                 out_size,
                 kernel_size_1,kernel_size_2,stride,padding,
                 dilation,p,
                 cls_num):
        super(CNNet,self).__init__()
        
        self.cls_num = cls_num
        self.embedding_size = embedding_size
        
        self.seq_len = seq_len
        
        self.out_size = out_size
        self.kernel_1 = kernel_size_1
        self.kernel_2 = kernel_size_2
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        
        #self.embedding = nn.Embedding(self.num_words+1, self.embedding_size, padding_idx = 0)
        
        
        self.conv1 = nn.Conv1d(self.seq_len,out_size,
                               kernel_size = self.kernel_1,
                               stride = self.stride,padding = self.padding,
                               dilation = self.dilation)
        
        self.pool1 = nn.MaxPool1d(self.kernel_1,self.stride)
        
        self.conv2 = nn.Conv1d(self.seq_len,out_size,
                               kernel_size = self.kernel_2,
                               stride = self.stride,padding = self.padding,
                               dilation = self.dilation)
        
        self.pool2 = nn.MaxPool1d(self.kernel_2,self.stride)
        
        self.linear = nn.Linear(self.in_feature(),cls_num)
        
        self.dropout = nn.Dropout(p)
        
        self.leaky = nn.LeakyReLU()
        self.softmax = nn.Softmax()
    
    def in_feature(self):
        
        '''Calculates the number of output features after Convolution + Max pooling
         
        Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        
        source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        '''
        # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
        out_conv_1 = ((self.embedding_size + (2*self.padding) - self.dilation * (self.kernel_1 - 1)-1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 + (2*self.padding) - self.dilation * (self.kernel_1 - 1)-1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)
        
        # Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
        out_conv_2 = ((self.embedding_size + (2*self.padding) - self.dilation * (self.kernel_2 - 1)-1) / self.stride) + 1
        out_conv_2 = math.floor(out_conv_2)
        out_pool_2 = ((out_conv_2 + (2*self.padding) - self.dilation * (self.kernel_2 - 1)-1) / self.stride) + 1
        out_pool_2 = math.floor(out_pool_2)
        

        
        return (out_pool_1+out_pool_2)*self.out_size
    
    def forward(self,x):
        
        #e = self.embedding(x)
        
        a1 = self.conv1(x)

        z1 = self.leaky(a1)
        
        x1 = self.pool1(z1)
        
        z2 = self.leaky(self.conv2(x))
        x2 = self.pool2(z2)
        
        
        union = torch.cat((x1,x2),2)
        
        union = union.reshape(union.size(0),-1)
        
        
        
        out = self.linear(union)
        if self.cls_num>1:
            print(out)
            out = F.softmax(out,dim = 1)
            print(out) 
        return out

    
def cnn_classifier(X_train, y_train, X_valid, y_valid, cnn_params, model_path):
    
    epochs = cnn_params['epochs']
    lr = cnn_params['learning_rate']
    eps = cnn_params['eps']
    batch_size = cnn_params['batch_size']
    
    out_size = cnn_params['out_size']
    kernel_size_1 = cnn_params['kernel_size_1']
    kernel_size_2 = cnn_params['kernel_size_2']
    stride = cnn_params['stride']
    padding = cnn_params['padding']
    dilation = cnn_params['dilation']
    p = cnn_params['p']
    
    nfold = cnn_params['nfold']

    if isinstance(X_train,(np.ndarray,list)):
        
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)

        if X_valid is not None:
            X_valid = torch.FloatTensor(X_valid)
            y_valid = torch.FloatTensor(y_valid)

    if len(X_train.size()) == 2:
        X_train = X_train.unsqueeze(1)
        n = 1
        
        if X_valid is not None:
            X_valid = X_valid.unsqueeze(1)
    else:
        n = len(X_train[0])

    
    if y_train.unsqueeze(1).size()[-1]>1:
        
        model = CNNet(len(X_train[0][0]),n,out_size,kernel_size_1,kernel_size_2,stride,padding,
                     dilation,p,y_train.unsqueeze(1).size()[-1]).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        model = CNNet(len(X_train[0][0]),n,out_size,kernel_size_1,kernel_size_2,stride,padding,
                     dilation,p).to(device)
    
        criterion = nn.BCEWithLogitsLoss()
        
        y_train = y_train.unsqueeze(1)
        y_valid = y_valid.unsqueeze(1)
    
    
    if model_path is not None: 
        if os.path.isfile(model_path):
            print('model already exists, it is loading..')
            model.load_state_dict(torch.load(model_path))
            return model
        
        
    
    optim = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=eps)
    
    
    if X_valid is not None:
        
        best_loss = float('inf')
        train_batch_loader = batch_loader(X_train,y_train,batch_size)
        valid_batch_loader = batch_loader(X_valid,y_valid,batch_size)
        
        for epoch in range(epochs):
            
            train_loss = 0.0
            valid_loss = 0.0
            
            model.train()
            for x_tr,y_tr in train_batch_loader:
                
                optim.zero_grad()
                pred = model(x_tr.to(device).float())
                loss = criterion(pred,y_tr.to(device).float())
                train_loss += float(loss.item())
                loss.backward()
                optim.step()

            model.eval()
            with torch.no_grad():
                for x_v,y_v in valid_batch_loader:
                    pred_test = model(x_v.to(device).float())
                    loss = criterion(pred_test,y_v.to(device).float())
                    valid_loss+=loss.item()
            
            if best_loss>valid_loss:
                best_loss = valid_loss
                best_model = model
            
            print(f'epoch: {epoch+1} valid loss: {valid_loss}, train_loss: {train_loss}')
            if train_loss<best_loss:break 
        
        if model_path is not None:
            print('saving model..')
            torch.save(best_model.state_dict(),model_path)
        return best_model

            
    else:
        rkf = RepeatedKFold(n_splits = nfold, n_repeats = 2, random_state = 10000)
        best_loss = float('inf')
        for epoch in range(epochs):
            
            train_loss = 0.0
            valid_loss = 0.0
            
            for train_idx,valid_idx in rkf.split(X_train):
                train_batch_loader = batch_loader(X_train[train_idx],y_train[train_idx],batch_size)
                valid_batch_loader = batch_loader(X_train[valid_idx],y_train[valid_idx],batch_size)
                model.train()
                for x_tr,y_tr in train_batch_loader:
                    
                    optim.zero_grad()
                    pred = model(x_tr.to(device).float())
                    loss = criterion(pred,y_tr.to(device).float())
                    train_loss += float(loss.item())
                    loss.backward()
                    optim.step()
                
                
                model.eval()
                with torch.no_grad():
                    for x_v,y_v in valid_batch_loader:
                        pred_test = model(x_v.to(device).float())
                        loss = criterion(pred_test,y_v.to(device).float())
                        valid_loss+=loss.item()
                

            print(f'epoch: {epoch+1} valid loss: {valid_loss}, train_loss: {train_loss}')
            if best_loss>valid_loss:
                best_loss = valid_loss
                best_model = model
        
            if train_loss<best_loss:break
        
        if model_path is not None:
            print('saving model..')
            torch.save(best_model.state_dict(),model_path)
        return best_model
    



