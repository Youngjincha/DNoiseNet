# -*- coding: utf-8 -*-
"""
@author: Alireza
"""

from IPython.display import display

display()
#%% import libraries
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

#CUDA_LAUNCH_BLOCKING=1

torch.cuda.empty_cache()

import pickle
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import rir_generator as rir
from scipy import signal

#%% Import data (48 kHz)

path = "./Data/48kHz/"

samplerate, train_data = wavfile.read(path+'train_BGH.wav')
samplerate, test_data = wavfile.read(path+'test_BGH.wav')
test_data = test_data.reshape((len(test_data),1))
train_data = train_data.reshape((len(train_data),1))
plt.plot(train_data)
plt.plot(test_data)


#%% import data

scaler = MinMaxScaler(feature_range=(-1, 1))
path = "./Data/testing/"
test_data = np.loadtxt(path+"aero2_testing_0404.csv", dtype=np.float64)
test_data =scaler.fit_transform( test_data.reshape((len(test_data),1)))

path = "./Data/training/"
train_data = np.loadtxt(path+"aero2_training_0404.csv", dtype=np.float64)
train_data =scaler.fit_transform( train_data.reshape((len(train_data),1)))
plt.plot(train_data)
plt.plot(test_data)

#%% 88.2 Khz data

path = 'C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/GPU_2022_6_6/Previous paper/pytorch/'
train_data = np.loadtxt(path+"train_data_2.csv", dtype=np.float64)
train_data = train_data.reshape((len(train_data),1))
test_data = np.loadtxt(path+"test_data_2.csv", dtype=np.float64)
test_data = test_data.reshape((len(test_data),1))

scaler = MinMaxScaler(feature_range=(0, 1))

plt.plot(train_data)
plt.plot(test_data)

#%% Create measurement noise
sc1  = 390000
sc2  = 390000
white_noise_train1 = np.random.normal(loc = 0, scale = 1, size = (len(train_data),1))
white_noise_train2 = np.random.normal(loc = 0, scale = 1, size = (len(train_data),1))

white_noise_train1 = white_noise_train1/sc1
white_noise_train2 = white_noise_train2/sc2
print('SNR train data',10*math.log10(np.sum(train_data**2)/np.sum(white_noise_train1**2)))
print('SNR train data',10*math.log10(np.sum(train_data**2)/np.sum(white_noise_train2**2)))

plt.plot(white_noise_train2)

#for test data
white_noise_test1 = np.random.normal(loc = 0, scale = 1, size = (len(test_data),1))
white_noise_test1 = white_noise_test1/sc1
white_noise_test2 = np.random.normal(loc = 0, scale = 1, size = (len(test_data),1))
white_noise_test2 = white_noise_test2/sc2

print('SNR test data',10*math.log10( np.sum(test_data**2)/np.sum(white_noise_test1**2)))
print('SNR test data',10*math.log10( np.sum(test_data**2)/np.sum(white_noise_test2**2)))

plt.plot(white_noise_test2)

white_noise_train_tns = torch.from_numpy(white_noise_train1.astype(np.float32)).to(device)
white_noise_test_tns = torch.from_numpy(white_noise_test1.astype(np.float32)).to(device)

#%% produce RIRs

h_s = rir.generate(
    c=340,                  # Sound velocity (m/s)
    fs=48000,                  # Sample frequency (samples/s)
    r=[                     # Receiver position(s) [x y z] (m)
        [1.5, 2.5, 1]
    ],
    s=[1.5, 3, 1],          # Source position [x y z] (m)
    dim = 3,
    L=[3, 4, 2],            # Room dimensions [x y z] (m)
    reverberation_time=0.2, # Reverberation time (s)
    nsample=1500,           # Number of output samples
    mtype= rir.mtype.omnidirectional, # Type of microphone
    order = 3,
    hp_filter = False
)


h_p = rir.generate(
    c=340,                  # Sound velocity (m/s)
    fs=48000,                  # Sample frequency (samples/s)
    r=[                     # Receiver position(s) [x y z] (m)
        [1.5, 2.5, 1]
    ],
    s=[1.5, 1, 1],          # Source position [x y z] (m)
    dim = 3,
    L=[3, 4, 2],            # Room dimensions [x y z] (m)
    reverberation_time=0.2, # Reverberation time (s)
    nsample=1500,           # Number of output samples
    mtype= rir.mtype.omnidirectional, # Type of microphone
    order = 3,
    hp_filter = False
)

h_p = h_p/np.linalg.norm(h_p)
h_s = h_s/np.linalg.norm(h_s)

h_s_tns = torch.from_numpy(h_s.astype(np.float32)).to(device)

plt.figure(figsize=(12,8))
plt.plot(h_p, label='h_p')
plt.plot(h_s, label='h_s')
plt.legend(loc="upper right",fontsize='large')
plt.title('Room Impulse Response, reverberation_time=0.2')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

x = train_data[:,0]

Convolved_x_p = np.convolve( x, h_p[:,0],mode = 'same')
Convolved_x_s = np.convolve(h_s[:,0], x, mode = 'same')
print('Noise Attenuation Primary path',10*math.log10(np.sum(x**2)/np.sum(Convolved_x_p**2)))
print('Noise Attenuation secondary path',10*math.log10(np.sum(x**2)/np.sum(Convolved_x_s**2)))

plt.figure(figsize=(12,8))
plt.plot(x, label= 'reference mic')
plt.plot(Convolved_x_p, label= 'error mic')
plt.legend(loc="upper right",fontsize='large')
plt.title('Colvolved X with h_p')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
#%%"""**Creating dataset**""" for filtered data

def input_output_generator_LSTM(z,n):
    #generating input output sets for LSTM models
    x_train = []
    y_train = []
    z_train = z[:math.floor(trainfrac*len(z))]
    for i in range(len(z_train)-n):
        x_train.append(z[i:i+n])
        y_train.append(z[i+n])
   
    x_train = np.array(x_train)
    x_train_lstm = np.reshape(x_train,(x_train.shape[0], 1,x_train.shape[1]))
    return x_train_lstm, np.array(y_train)#, x_test_lstm, np.array(y_test)

trainfrac = 1
lag_sn = 20
LB = lag_sn

d_train = signal.lfilter(h_p[:,0], 1, train_data[:,0]).reshape(len(train_data),1)
d_test = signal.lfilter(h_p[:,0], 1, test_data[:,0]).reshape(len(test_data),1)


X_noise_train,Y_noise_train = input_output_generator_LSTM(d_train+white_noise_train1, lag_sn)
X_noise_test,Y_noise_test = input_output_generator_LSTM(d_test+white_noise_test1, lag_sn)

a,Y_noise_train = input_output_generator_LSTM(d_train+white_noise_train2, lag_sn)
a,Y_noise_test = input_output_generator_LSTM(d_test+white_noise_test2, lag_sn)


np.max(X_noise_train)
np.min(X_noise_train)
np.max(X_noise_test)

X_noise_train_tns = torch.from_numpy(X_noise_train.astype(np.float32))
Y_noise_train_tns = torch.from_numpy(Y_noise_train.astype(np.float32))
X_noise_test_tns = torch.from_numpy(X_noise_test.astype(np.float32))
Y_noise_test_tns = torch.from_numpy(Y_noise_test.astype(np.float32))


plt.figure(figsize=(12,8))
plt.plot(d_train, label= 'reference mic')
plt.plot(Y_noise_train, label= 'error mic')
plt.legend(loc="upper right",fontsize='large')
plt.title('Colvolved X with h_p')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.figure(figsize=(12,8))
plt.plot(test_data, label= 'reference mic')
plt.plot(Y_noise_test, label= 'error mic')
plt.legend(loc="upper right",fontsize='large')
plt.title('Colvolved X with h_p')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

#%%
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.dense1 = torch.nn.Linear(in_features = 20, out_features = 20, bias=True, device=None, dtype=None)
        self.out = torch.nn.Linear(in_features=20, out_features = 1, bias=True, device=None, dtype=None)
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        
        l1 = self.tanh(self.dense1(x))
        out = self.out(l1)
        return out



# torch.cuda.empty_cache()
mynetwork = MLP()
batch_size = 100
# CUDA_LAUNCH_BLOCKING=1
print('Model inputs')
print(summary2(mynetwork.cpu(), torch.zeros((batch_size,20)), show_input=True))
print('Model Outputs')
print(summary2(mynetwork.cpu(), torch.zeros((batch_size,20)), show_input=False))

mynetwork = MLP().to(device)
#%% Loss and optimizer
class custom_loss(nn.Module):
    def __init__(self, weight=None, size_average=False):
        super(custom_loss, self).__init__()
        #MSE = nn.MSELoss() 
        
    def forward(self, output, targets):
       
        #F_01 = 0.22360679774*torch.sqrt(torch.tensor((math.pi)))*torch.erf(2.2360679774*inputs)

        F_05 = 0.5*1.772453*torch.erf(output)
        output2 = torch.sum(torch.matmul(F_05.view(F_05.shape[0],1),torch.transpose(h_s_tns, 0,1)),1)
  
       
        return torch.mean(torch.pow(targets[:,0]-output2[:], 2))
    

loss_fn = custom_loss().to(device)
criterion = loss_fn

optimizer = torch.optim.AdamW(mynetwork.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.01, amsgrad=False)
num_epochs = 20
#scheduler = PolynomialLRDecay(optimizer, max_decay_steps=num_epochs, end_learning_rate=0.00001, power=0.9)
#%%
train = torch.utils.data.TensorDataset(X_noise_train_tns.to(device), Y_noise_train_tns.to(device))
test = torch.utils.data.TensorDataset(X_noise_test_tns.to(device), Y_noise_test_tns.to(device))

train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=True)

batch_size = 100
n_iterations = math.ceil(len(X_noise_train)/batch_size)
print(n_iterations)

validation_iteration = math.ceil(len(X_noise_test)/batch_size)

#PATH = './MLP_mix_FW_ON_FB_PS_eta_0.5_autosave1.pth'

epoch_loss_saved = []
valid_loss_saved = []
since = time.time()
for epoch in range(num_epochs):
    train_loss = 0.0
    mynetwork.train()
    for step, (X, Y) in enumerate(train_loader): 
        # Forward pass
        # Clear the gradients
        optimizer.zero_grad()
        outputs = mynetwork(X)
        loss = criterion(outputs, Y)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        train_loss += loss.item() 
        
        
    #scheduler.step(epoch)
    mynetwork.eval() 
    valid_loss = 0.0
    with torch.no_grad():
        for step, (X_val, Y_val) in enumerate(test_loader):
          # Forward Pass
          pred_val = mynetwork(X_val)
          # Find the Loss
          loss = criterion(pred_val,Y_val)
          # Calculate Loss
          valid_loss += loss.item()
          
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / n_iterations} \t\t Validation Loss: {valid_loss / validation_iteration}')
   # print(outputs[0].item(),Y[0].item())
    epoch_loss_saved.append(train_loss / n_iterations)
    valid_loss_saved.append(valid_loss / validation_iteration)

    if valid_loss_saved[epoch] <= np.min(valid_loss_saved):
        #torch.save(mynetwork, PATH)
        print('model saved')
 
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

#%%  saving model

PATH = './MLP_mix_FW_ON_FB_PS_eta_0.5_1.pth'

torch.save(mynetwork, PATH)

#%% plot the training and test loss
plt.figure(figsize=(10, 5))
plt.plot(epoch_loss_saved, "-b", label="Training Loss")
plt.plot(valid_loss_saved, "-r", label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(loc="upper right",fontsize='large')
plt.show()


#%%

X_test = X_noise_test_tns.to(device)

mynetwork.eval() 
since = time.time()
with torch.no_grad():
      pred_test = mynetwork(X_test)
time_elapsed = time.time() - since
#print(f'7.5 second of data : {time_elapsed} s' )

pred_test1 = 0.5*1.772453*torch.erf(pred_test)
pred_test2_1 = torch.sum(torch.matmul(pred_test1.view(pred_test1.shape[0],1),torch.transpose(h_s_tns, 0,1)),1)
pred_test2 = pred_test2_1.cpu().numpy().reshape((len(pred_test2_1),1))

error = Y_noise_test[:,0]-pred_test2[:,0]
error = error - np.mean(error)
Primary_noise = Y_noise_test - np.mean(Y_noise_test)
plt.figure(figsize=(12, 8))
plt.plot(Primary_noise, label='Primary noise')
plt.plot(error,label = 'Attenuated noise')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend(loc="upper right",fontsize='large')
plt.show()

noise_attenuation = 10*math.log10(np.sum(Primary_noise**2)/np.sum(error**2))
print('Noise Attenuation= ',noise_attenuation)

np.min(pred_test2)
np.max(pred_test2)


#%% load MLP network
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

PATH_mlp = './MLP7_autosave.pth'
MLP = torch.load(PATH_mlp)

#%%

class custom_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(custom_loss, self).__init__()
       
    def forward(self, output, targets):
        
        F_05 = 0.5*1.772453*torch.erf(output)
        output2 = torch.sum(h_s_tns*F_05)
        
        return targets-output2


loss_fn = custom_loss().to(device)

criterion = loss_fn
#%% Testing on feedback 
Xh = torch.zeros(1).to(device)
loss0 = torch.zeros(1).to(device)
outputs0 = torch.zeros(1).to(device)
X_val = X_noise_test_tns[0:1, :, :]
valid_loss = 0.0

validation_iteration = math.ceil(len(X_noise_test))
print(validation_iteration)

pred_test = np.zeros(X_noise_test.shape[0])
mynetwork.eval()
with torch.no_grad():
    for i in range(0, validation_iteration):
        # print(X_val)
        X_val = X_val.to(device)
        Y_val = Y_noise_test_tns[i]
        Y_val = Y_val.to(device)

        pred_val = mynetwork(X_val)
        pred_test[i] = pred_val.item()
        loss = criterion(pred_val, Y_val) + white_noise_test_tns[i]
        #print(pred_val, Y_val)
        loss0[:] = loss
        outputs0[:] = pred_val
        
        #F_05 = 0.5*1.772453*torch.erf(outputs0)
        #outputs0 = F_05
        outputs0 = MLP(outputs0.view(1,1))
        #Xh = ((Y_val >= outputs0)*(torch.sqrt(loss0)+outputs0) +(Y_val < outputs0)*(-torch.sqrt(loss0)+outputs0))
        #Xh = ((Y_val >= outputs0)*(torch.sqrt(loss0)) +(Y_val < outputs0)*(-torch.sqrt(loss0)))
        Xh = loss0 + outputs0
        loss = torch.pow(loss,2)
        
        X_val[0:1, :, 0:LB-1] = X_val[0:1, :, 1:LB].clone()
        X_val[0:1, :, LB-1] = Xh[0]

        loss = torch.mean(loss)
        valid_loss += loss.item()
        #print(i)
print(f'Test Loss: {valid_loss / validation_iteration}')

#pred_test1 = 0.396332557*torch.erf(2.2360679774*torch.tensor(pred_test))
pred_test1 = 0.5*1.772453*torch.erf(torch.tensor(pred_test)).view(len(pred_test),1)
pred_test1 = torch.sum(torch.matmul(pred_test1,torch.transpose(torch.tensor(h_s), 0,1)),1)

pred_test1 = pred_test1.numpy()

error = Y_noise_test[:,0]-pred_test1[:]
error = error - np.mean(error)
Primary_noise = Y_noise_test - np.mean(Y_noise_test)
plt.figure(figsize=(12, 8))
plt.plot(Primary_noise, label = 'Primary noise')
plt.plot(error, label = 'Attenuated noise')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('Test data')
plt.legend(loc="upper right",fontsize='large')
plt.show()

noise_attenuation = 10*math.log10(np.sum(Primary_noise**2)/np.sum(error**2))
print('Noise Attenuation= ',noise_attenuation)
#%% saving the results
DataName= 'MLP_FB_MLP_const3_test_0.5.csv'
np.savetxt("path"+DataName, Y_noise_test, delimiter=",")

DataName= 'MLP_FB_MLP_const3_pred_test_0.5.csv'
np.savetxt("path"+DataName, pred_test1, delimiter=",")



