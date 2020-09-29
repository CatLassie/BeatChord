#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

import torch
# import torchvision

import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[ ]:


# model path
CURRENT_PATH = os.getcwd()
MODEL_PATH = os.path.join(CURRENT_PATH, 'models')

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)  
    
MODEL_NAME = 'model'


# In[ ]:


# import data
mnist_train = pd.read_csv('../../../datasets/mnist_csv/mnist_train.csv')
mnist_test = pd.read_csv('../../../datasets/mnist_csv/mnist_test.csv')


# In[ ]:


# prepare data
train = mnist_train.dropna()
train_feat = mnist_train.drop('label', axis=1)
train_target = mnist_train['label']

test = mnist_test.dropna()
test_feat = mnist_test.drop('label', axis=1)
test_target = mnist_test['label']


# In[ ]:


# convert to tensors
train_f = torch.tensor(train_feat.values, dtype=torch.float)
train_t = torch.tensor(train_target.values, dtype=torch.long)
test_f = torch.tensor(test_feat.values, dtype=torch.float)
test_t = torch.tensor(test_target.values, dtype=torch.long)
train_f = train_f.reshape(-1,1,28,28)
test_f = test_f.reshape(-1,1,28,28)
print(train_f.shape)
print(train_t.shape)
print(test_f.shape)
print(test_t.shape)


# In[ ]:


# network params

# depth
in_size = 1
h1_size = 16
h2_size = 32

# kernel size
k_conv_size = 5
k_pool_size = 2

# fully connected parameters
fc_size = 512
out_size = 10

#number of epochs
num_epochs = 10

# learning rate
lr = 0.001

# loss function
loss_func = nn.CrossEntropyLoss()


# In[ ]:


class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.l1 = nn.Sequential(
            nn.Conv2d(in_size, h1_size, k_conv_size),
            nn.BatchNorm2d(h1_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = k_pool_size)
        )
        
        self.l2 = nn.Sequential(
            nn.Conv2d(h1_size, h2_size, k_conv_size),
            nn.BatchNorm2d(h2_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = k_pool_size)
        )
        
        self.fc = nn.Linear(fc_size, out_size)
    
    def forward(self, x):
        out = self.l1(x)
        # print(out.shape)
        
        out = self.l2(out)
        # print(out.shape)
        
        out = out.reshape(out.size(0), -1)
        # print(out.shape)
        
        out = self.fc(out)
        # print(out.shape)
        
        return out


# In[ ]:


# model
model = ConvNet()

# optimizer
opt = torch.optim.Adam(model.parameters(), lr=lr)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CURRENT DEVICE:", device)


# In[ ]:


model.to(device)


# In[ ]:


train_f = train_f.to(device)
train_t = train_t.to(device)
test_f = test_f.to(device)
test_t = test_t.to(device)


# In[ ]:


def training():    
    # loss_values = list()

    for epoch in range(1, num_epochs):

        outputs = model(train_f)
        loss = loss_func(outputs, train_t)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print('Epoch - %d, loss - %0.5f '%(epoch, loss.item()))
        # loss_values.append(loss.item())


# In[ ]:


# train and save model
# training()
# torch.save(model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME + '.model'))


# In[ ]:


# load model
model = ConvNet().to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME + '.model')))


# In[ ]:


model.eval()

with torch.no_grad():
    
    correct = 0
    total = 0
    
    outputs = model(test_f)
    _, predicted = torch.max(outputs.data, 1)
    
    test_t_cpu = test_t.cpu().numpy()
    predicted = predicted.cpu()
    
    print("Accuracy", accuracy_score(predicted, test_t_cpu))
    print("Precision", precision_score(predicted, test_t_cpu, average='weighted'))
    print("Recall", recall_score(predicted, test_t_cpu, average='weighted'))

