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
from torch.utils.data import DataLoader


import math
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(torch.cuda.get_device_name(device))



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
                 ho):
        
        super(RNNet,self).__init__()
        
        self.embedding_size = embedding_size
        # self.num_words = num_words
        # self.seq_len = seq_len
        
        self.out_size = out_size
        self.num_layers = num_layers
        
        self.ho = ho
        
        #self.embedding = nn.Embedding(self.num_words+1, self.embedding_size, padding_idx = 0)
        
        self.rnn = nn.RNN(self.embedding_size, self.out_size, num_layers = self.num_layers,
                          dropout = p)
        
        self.linear = nn.Linear(self.out_size, 1)

        
    
    def forward(self,x):
        
        #embeddings = self.embedding(x)
        #print(x.size())
        output, hidden = self.rnn(x, self.ho)
        
        out = self.linear(output[:,-1])
        return out


def rnn_classifier(X_train, y_train, X_valid, y_valid, rnn_params, model_path):
    
    
    embedding_size = rnn_params['embedding_size']
    epochs = rnn_params['epochs']
    lr = rnn_params['learning_rate']
    eps = rnn_params['eps']
    batch_size = rnn_params['batch_size']
    
    out_size = rnn_params['out_size']
    p = rnn_params['p']
    num_layers = rnn_params['num_layers']
    
    
    nfold = rnn_params['nfold']
    
    #X_train,encoder = onehot_encoder(X_train, y_train)
    #if X_valid:
    #    X_valid = encoder.transform(X_valid)
    if isinstance(X_train,np.ndarray):
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        if X_valid is not None:
            X_valid = torch.from_numpy(X_valid)
            y_valid = torch.from_numpy(y_valid)
    
    #print(X_train.size())
    
    ho = torch.randn(num_layers, 1, out_size).to(device)
    
    model = RNNet(len(X_train[0]),out_size,
                 num_layers,p,ho).to(device)
    
    if os.path.isfile(model_path):
        print('model already exists, it is loading..')
        model.load_state_dict(torch.load(model_path))
        return model
    
    #model = RNNet(embedding_size,num_words,len(X_train[0]),out_size,
    #             num_layers,p,ho).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
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
                pred = model(x_tr.to(device).unsqueeze(1).float())
                loss = criterion(pred,y_tr.to(device).unsqueeze(1).float())
                train_loss += float(loss.item())
                loss.backward()
                optim.step()
            
            
            model.eval()
            with torch.no_grad():
                
                pred_test = model(X_valid.to(device).unsqueeze(1).float())
                loss = criterion(pred_test,y_valid.to(device).unsqueeze(1).float())
                valid_loss+=loss.item()
                
            print(f'epoch: {epoch+1} valid loss: {valid_loss}, train_loss: {train_loss}')
            
            if best_loss>valid_loss:
                best_loss = valid_loss
                best_model = model
            if train_loss<best_loss:break
        print('saving model..')
        torch.save(best_model.state_dict(),model_path)
        return best_model
            
    else:
        rkf = RepeatedKFold(n_splits = nfold, n_repeats = 2, random_state = 10000)
        best_loss = float('inf')
        for epoch in range(epochs):
            
            train_loss = 0.0
            valid_loss = 0.0
            
            #best_loss = float('inf')
            
            for train_idx,valid_idx in rkf.split(X_train):
                
                train_batch_loader = batch_loader(X_train[train_idx],y_train[train_idx],batch_size)
                
                model.train()
                for x_tr,y_tr in train_batch_loader:
                    
                    optim.zero_grad()
                    #print(x_tr.size())
                    pred = model(x_tr.to(device).float().unsqueeze(1))
                    loss = criterion(pred,y_tr.to(device).float().unsqueeze(1))
                    train_loss += float(loss.item())
                    loss.backward()
                    optim.step()
                
                
                model.eval()
                with torch.no_grad():
                    pred_test = model(X_train[valid_idx].to(device).float().unsqueeze(1))
                    loss = criterion(pred_test,y_train[valid_idx].to(device).float().unsqueeze(1))
                    valid_loss+=loss.item()
                

            if best_loss>valid_loss:
                best_loss = valid_loss
                best_model = model
            print(f'epoch: {epoch+1} valid loss: {valid_loss}, train_loss: {train_loss}')
            if train_Loss< best_loss:break
        
        print('saving model..')
        torch.save(best_model.state_dict(),model_path)
        return best_model
    

class CNNet(nn.Module):
    
    
    def __init__(self,embedding_size,seq_len,
                 out_size,
                 kernel_size_1,kernel_size_2,stride,padding,
                 dilation,p):
        super(CNNet,self).__init__()
        
        
        self.embedding_size = embedding_size
        # self.num_words = num_words
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
        
        self.linear = nn.Linear(self.in_feature(),1)
        
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
        
        #print(x2.size())
        union = torch.cat((x1,x2),2)
        #print(union.size())
        union = union.reshape(union.size(0),-1)
        #print(union.size())
        
        
        out = self.linear(union)

        
        return out

    
def cnn_classifier(X_train, y_train, X_valid, y_valid, cnn_params, model_path):
    
    embedding_size = cnn_params['embedding_size']
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
    
    #X_train,encoder = onehot_encoder(X_train, y_train)
    #if X_valid:
    #    X_valid = encoder.transform(X_valid)
    
    if isinstance(X_train,np.ndarray):
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        if X_valid is not None:
            X_valid = torch.from_numpy(X_valid)
            y_valid = torch.from_numpy(y_valid)
    
    
    model = CNNet(len(X_train[0]),1,out_size,kernel_size_1,kernel_size_2,stride,padding,
                 dilation,p).to(device)
    
    if os.path.isfile(model_path):
        print('model already exists, it is loading..')
        model.load_state_dict(torch.load(model_path))
        return model
    
    #model = CNNet(embedding_size,num_words,len(X_train[0]),out_size,
    #             kernel_size_1,kernel_size_2,stride,padding,
    #             dilation,p).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
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
                pred = model(x_tr.to(device).unsqueeze(1).float())
                loss = criterion(pred,y_tr.to(device).unsqueeze(1).float())
                train_loss += float(loss.item())
                loss.backward()
                optim.step()
            #print(f'epoch: {epoch} training loss {train_loss}')
            
            model.eval()
            with torch.no_grad():
                for x_v,y_v in valid_batch_loader:
                    pred_test = model(x_v.to(device).unsqueeze(1).float())
                    loss = criterion(pred_test,y_v.to(device).unsqueeze(1).float())
                    #pred_test = model(X_valid.to(device).unsqueeze(1).float())
                    #loss = criterion(pred_test,y_valid.to(device).unsqueeze(1).float())
                    valid_loss+=loss.item()
            
            #print(f'epoch: {epoch} validation loss {valid_loss}')
            
            if best_loss>valid_loss:
                best_loss = valid_loss
                best_model = model
            
            print(f'epoch: {epoch+1} valid loss: {valid_loss}, train_loss: {train_loss}')
            if train_loss<best_loss:break 
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
                #best_loss = float('inf')
                train_batch_loader = batch_loader(X_train[train_idx],y_train[train_idx],batch_size)
                valid_batch_loader = batch_loader(X_train[valid_idx],y_train[valid_idx],batch_size)
                model.train()
                for x_tr,y_tr in train_batch_loader:
                    
                    optim.zero_grad()
                    pred = model(x_tr.to(device).unsqueeze(1).float())
                    #print(pred.size())
                    #print(y_tr.to(device).unsqueeze(1).float().size())
                    loss = criterion(pred,y_tr.to(device).unsqueeze(1).float())
                    train_loss += float(loss.item())
                    loss.backward()
                    optim.step()
                
                
                model.eval()
                with torch.no_grad():
                    for x_v,y_v in valid_batch_loader:
                        #print(X_train[valid_idx].size()) 
                        pred_test = model(x_v.to(device).unsqueeze(1).float())
                        loss = criterion(pred_test,y_v.to(device).unsqueeze(1).float())
                        #pred_test = model(X_train[valid_idx].to(device).unsqueeze(1).float())
                        #loss = criterion(pred_test,y_train[valid_idx].to(device).unsqueeze(1).float())
                        valid_loss+=loss.item()
                

            print(f'epoch: {epoch+1} valid loss: {valid_loss}, train_loss: {train_loss}')
            if best_loss>valid_loss:
                best_loss = valid_loss
                best_model = model
        
            if train_loss<best_loss:break
        print('saving model..')
        torch.save(best_model.state_dict(),model_path)
        return best_model
    



