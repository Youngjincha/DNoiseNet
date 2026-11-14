# -*- coding: utf-8 -*-
"""
@author: Alireza
"""

# %% import libraries
from sympy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.io import wavfile
import pickle
import numpy as np
import math
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from torch.optim import lr_scheduler

from torchinfo import summary
from pytorch_model_summary import summary as summary2

torch.cuda.is_available()

device2 = torch.device('cuda:2')
device = torch.device('cuda')
print(device)

# CUDA_LAUNCH_BLOCKING=1
torch.cuda.empty_cache()
import rir_generator as rir

#%% create teest and training data
scaler = MinMaxScaler(feature_range=(-1, 1))

test_data = np.random.rand(50000).reshape(50000,1)
train_data = np.random.rand(200000).reshape(200000,1)

train_data = scaler.fit_transform(train_data)
test_data = scaler.fit_transform(test_data)

plt.figure(figsize=(10, 5))
plt.plot(train_data, "-b", label="Train")
plt.plot(test_data, "-r", label="Test")
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend(loc="upper right",fontsize='large')
plt.show()

#%% genaring measurement noise (white noise)

sc  = 5500
white_noise_train1 = np.random.normal(loc = 0, scale = 1, size = (len(train_data),1))
white_noise_train2 = np.random.normal(loc = 0, scale = 1, size = (len(train_data),1))

white_noise_train1 = white_noise_train1/sc
white_noise_train2 = white_noise_train2/sc
print('SNR train data',10*math.log10(np.sum(train_data**2)/np.sum(white_noise_train1**2)))
print('SNR train data',10*math.log10(np.sum(train_data**2)/np.sum(white_noise_train2**2)))

plt.plot(white_noise_train1)

#for test data
white_noise_test1 = np.random.normal(loc = 0, scale = 1, size = (len(test_data),1))
white_noise_test1 = white_noise_test1/sc
white_noise_test2 = np.random.normal(loc = 0, scale = 1, size = (len(test_data),1))
white_noise_test2 = white_noise_test2/sc

print('SNR test data',10*math.log10( np.sum(test_data**2)/np.sum(white_noise_test1**2)))
print('SNR test data',10*math.log10( np.sum(test_data**2)/np.sum(white_noise_test2**2)))

plt.plot(white_noise_test1)


white_noise_train_tns = torch.from_numpy(white_noise_train1.astype(np.float32)).to(device)
white_noise_test_tns = torch.from_numpy(white_noise_test1.astype(np.float32)).to(device)

#%%"""**Creating dataset**""" for filtered data


h_s = rir.generate(
    c=340,                  # Sound velocity (m/s)
    fs=19980,                 # Sample frequency (samples/s)
    r=[                     # Receiver position(s) [x y z] (m)
        [1.5, 3, 1]
    ],
    s=[1.5, 2.5, 1],          # Source position [x y z] (m)
    dim = 3,
    L=[3, 4, 2],            # Room dimensions [x y z] (m)
    reverberation_time=0.2, # Reverberation time (s)
    nsample=1500,           # Number of output samples
    mtype= rir.mtype.omnidirectional, # Type of microphone
    order = 3,
    hp_filter = False
)

h_s = h_s/np.linalg.norm(h_s)

plt.figure(figsize=(12,8))
plt.plot(h_s, label='h_s')
plt.legend(loc="upper right",fontsize='large')
plt.title('Room Impulse Response, reverberation_time=0.2')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

#%%
X_noise_train = train_data
X_noise_test = test_data

def create_data(x):
    n = len(x)
    Y = np.zeros((n,1))
    for i in range(n):
        Y[i] = 0.5*1.772453*math.erf(x[i,0])
        Y[i] = np.sum(Y[i]*h_s)
    return Y

Y_noise_train = create_data(train_data+white_noise_train1)
Y_noise_test = create_data(test_data+white_noise_test1)

X_noise_train_tns = torch.from_numpy(X_noise_train.astype(np.float32))
Y_noise_train_tns = torch.from_numpy(Y_noise_train.astype(np.float32))
X_noise_test_tns = torch.from_numpy(X_noise_test.astype(np.float32))
Y_noise_test_tns = torch.from_numpy(Y_noise_test.astype(np.float32))


plt.figure(figsize=(12,8))
plt.legend(loc="upper right",fontsize='large')
plt.plot(train_data, label= 'reference mic')
plt.plot(Y_noise_train, label= 'error mic')
plt.title('Colvolved X with h_p')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
# %%

#"""**Creating MLP  1 2 3 module**"""
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.layer1 = nn.Linear(in_features=1, out_features=5, bias=True)
        self.layer2 = nn.Linear(in_features=5, out_features=3, bias=True)
        self.layer3 = nn.Linear(in_features=3, out_features=1, bias=True)
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.tanh(self.layer2(layer1))
        out = self.sig(self.layer3(layer2))
        
        return out
# torch.cuda.empty_cache()
mynetwork = MLP()
batch_size = 100
# CUDA_LAUNCH_BLOCKING=1
print('Model inputs')
print(summary2(mynetwork.cpu(), torch.zeros((batch_size, 1)), show_input=True))
print('Model Outputs')
print(summary2(mynetwork.cpu(), torch.zeros((batch_size, 1)), show_input=False))

mynetwork = MLP().to(device)
#%%
#"""**Creating MLP 4 and 5 module**"""
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.layer1 = nn.Linear(in_features=1, out_features=8, bias=True)
        self.layer2 = nn.Linear(in_features=8, out_features=4, bias=True)
        self.layer3 = nn.Linear(in_features=4, out_features=2, bias=True)
        self.layer4 = nn.Linear(in_features=2, out_features=1, bias=True)
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.tanh(self.layer2(layer1))
        layer3 = self.tanh(self.layer3(layer2))
        out = self.layer4(layer3)
        
        return out

torch.cuda.empty_cache()
mynetwork = MLP()
batch_size = 100
# CUDA_LAUNCH_BLOCKING=1
print('Model inputs')
print(summary2(mynetwork.cpu(), torch.zeros((batch_size, 1)), show_input=True))
print('Model Outputs')
print(summary2(mynetwork.cpu(), torch.zeros((batch_size, 1)), show_input=False))

mynetwork = MLP().to(device)
# %% Loss and optimizer
MSE = nn.MSELoss()
criterion = MSE
optimizer = torch.optim.AdamW(mynetwork.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.01, amsgrad=False)
num_epochs = 300

# %% training
n_iterations = math.ceil(len(X_noise_train)/batch_size)
print(n_iterations)

validation_iteration = math.ceil(len(X_noise_test)/batch_size)

PATH = './MLP8_autosave.pth'

iter_loss_saved = []
epoch_loss_saved = []
valid_loss_saved = []

since = time.time()
for epoch in range(num_epochs):
    train_loss = 0.0
    mynetwork.train()
    for i in range(n_iterations):

        X = X_noise_train_tns[batch_size*i:batch_size*(i+1),:]
        X = X.to(device)
        Y = Y_noise_train_tns[batch_size*i:batch_size*(i+1)]
        Y = Y.to(device)

        # Forward pass
        # Clear the gradients
        optimizer.zero_grad()
        outputs = mynetwork(X)
        loss = criterion(outputs, Y)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        train_loss += loss.item() 
        iter_loss_saved.append(loss.item())
        
    
    #scheduler.step(epoch)
    mynetwork.eval() 
    valid_loss = 0.0
    with torch.no_grad():
        for i in range(int(validation_iteration)):

          # Forward Pass
          X_val = X_noise_test_tns[batch_size*i:batch_size*(i+1),:]
          X_val = X_val.to(device)
          Y_val = Y_noise_test_tns[batch_size*i:batch_size*(i+1),:]
          Y_val = Y_val.to(device)

          pred_val = mynetwork(X_val)
          # Find the Loss
          loss = criterion(pred_val,Y_val)
          # Calculate Loss
          valid_loss += loss.item()
          
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / n_iterations} \t\t Validation Loss: {valid_loss / validation_iteration}')
    epoch_loss_saved.append(train_loss / n_iterations)
    valid_loss_saved.append(valid_loss / validation_iteration)

    if valid_loss_saved[epoch] <= np.min(valid_loss_saved):
        torch.save(mynetwork, PATH)
        print('model saved')
 
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

#%% Plot the training and validation loss
# MLP6 is for 19980*4 sampling frequency
# MLP7 is for 44100 sampling frequency
# MLP8 is for 19980 sampling frequency

plt.figure(figsize=(10, 5))
plt.plot(epoch_loss_saved, "-b", label="Training Loss")
plt.plot(valid_loss_saved, "-r", label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(loc="upper right",fontsize='large')
plt.show()

#%%  saving model
PATH = './MLP4.pth'
torch.save(mynetwork, PATH)

#%% test the model
valid_loss = 0.0

validation_iteration = math.ceil(len(X_noise_test))
print(validation_iteration)

pred_test = np.zeros(X_noise_test.shape[0])
mynetwork.eval() 
with torch.no_grad():
    for i in range(0,validation_iteration):
        X_val = X_noise_test_tns[i:i+1,:]
        X_val = X_val.to(device)
        Y_val = Y_noise_test_tns[i]
        Y_val = Y_val.to(device)

        pred_val = mynetwork(X_val)
        pred_test[i] = pred_val.item()
        loss = criterion(pred_val,Y_val)
        
        loss = torch.mean(loss)
        valid_loss += loss.item()
print(f' Test Loss: {valid_loss / validation_iteration}')        

error = Y_noise_test[:,0]-pred_test[:]

plt.figure(figsize=(12, 8))
plt.plot(Y_noise_test, "-b", label="Test")
plt.plot(pred_test, "-g", label="Pred test")
plt.plot(error, "-r", label="Error")
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend(loc="upper right",fontsize='large')
plt.show()


