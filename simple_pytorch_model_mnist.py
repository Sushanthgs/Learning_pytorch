# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 11:59:59 2021

@author: sushanthsgradlaptop2
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
#%%
batch_size=64
training_data=datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
        )
test_data=datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
        )
#%%
train_dataloader=DataLoader(training_data,batch_size=batch_size)
test_dataloader=DataLoader(test_data,batch_size=batch_size)
for X,y in test_dataloader:
    print('Shape of X:',X.shape)
    print('Shape of Y:',y.shape)
    break
#%%
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten=nn.Flatten()
        self.linear_Relu_stack=nn.Sequential(nn.Linear(784,128),
                                             nn.ReLU(),
                                             nn.Linear(128,64),
                                             nn.ReLU(),
                                             nn.Linear(64,10),
                                             nn.Softmax())
    def forward(self,x):
        x=self.flatten(x)
        act_vals=self.linear_Relu_stack(x)
        return(act_vals)
model=NeuralNetwork()
print(model)
#%%
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
#%%
def train(dataloader,model,loss_fn,optimizer):
    size=len(dataloader.dataset)
    model.train()
    for batch,(X,y) in enumerate(dataloader):
        pred=model(X)
        loss=loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(batch % 100==0):
            loss, current=loss.item(),batch*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
            
    
        
    