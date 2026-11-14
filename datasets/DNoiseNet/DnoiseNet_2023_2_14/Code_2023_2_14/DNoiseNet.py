# -*- coding: utf-8 -*-
"""
@author: Alireza Mostafavi and Youngjin Cha
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
    x_train = []
    y_train = []
    z_train = z[:math.floor(trainfrac*len(z))]
    for i in range(len(z_train)-n):
        x_train.append(z[i:i+n])
        y_train.append(z[i+n])

    x_train = np.array(x_train)
    x_train_lstm = np.reshape(x_train,(x_train.shape[0], 1,x_train.shape[1]))
    return x_train_lstm, np.array(y_train)

trainfrac = 1
lag_sn = 20
LB = lag_sn

#%% without primary path

X_noise_train,_ = input_output_generator_LSTM(train_data, lag_sn)
X_noise_test,_ = input_output_generator_LSTM(test_data, lag_sn)

_,Y_noise_train = input_output_generator_LSTM(train_data+white_noise_train2, lag_sn)
_,Y_noise_test = input_output_generator_LSTM(test_data+white_noise_test2, lag_sn)


#%% conv using signal.filt
d_train = signal.lfilter(torch.from_numpy(h_p[:,0]), 1, torch.from_numpy(train_data[:,0])).reshape(len(train_data),1)
d_test = signal.lfilter(torch.from_numpy(h_p[:,0]), 1, torch.from_numpy(test_data[:,0])).reshape(len(test_data),1)

X_noise_train, Y_noise_train = input_output_generator_LSTM(train_data[0:len(d_train)]+white_noise_train1[0:len(d_train)], lag_sn)
X_noise_test, Y_noise_test = input_output_generator_LSTM(test_data[0:len(d_test)]+white_noise_test1[0:len(d_test)], lag_sn)

a,Y_noise_train = input_output_generator_LSTM(d_train+white_noise_train2[0:len(d_train)], lag_sn)
a,Y_noise_test = input_output_generator_LSTM(d_test+white_noise_test2[0:len(d_test)], lag_sn)
a=0
#%% transfer data from numpy to torch
X_noise_train_tns = torch.from_numpy(X_noise_train.astype(np.float32))
Y_noise_train_tns = torch.from_numpy(Y_noise_train.astype(np.float32))
X_noise_test_tns = torch.from_numpy(X_noise_test.astype(np.float32))
Y_noise_test_tns = torch.from_numpy(Y_noise_test.astype(np.float32))

plt.figure(figsize=(12,8))
plt.plot(train_data, label= 'reference mic')
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

#%% build the model 

class DCCNN_Block(nn.Module):
    def __init__(self, n_filters, filter_width, dilation_rate):
        super(DCCNN_Block, self).__init__()

        self.layer_1 = nn.Conv1d(in_channels=1, out_channels=n_filters,
                                 kernel_size=filter_width, dilation=dilation_rate, padding='same', bias=False)
        self.skip = nn.Conv1d(n_filters, 1, kernel_size=1,
                              padding='same', bias=False)
        self.layer_2 = nn.Conv1d(in_channels=n_filters, out_channels=1,
                                 kernel_size=1, dilation=1, padding='same', bias=False)
        self.selu = nn.SELU(inplace=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        layer_1 = self.layer_1(x)
        layer_1 = self.selu(layer_1)
        skip = self.skip(layer_1)
        layer_2 = self.layer_2(layer_1)
        To_Next_Layer = torch.add(x, layer_2)

        return To_Next_Layer, skip


class Model_Build_SN(nn.Module):
    def __init__(self):
        super(Model_Build_SN, self).__init__()
        dilation_rates = [2**i for i in range(7)]
        self.laa = nn.Conv1d(1, 1, kernel_size=3, padding='same', bias=True)
        self.DC_block1 = DCCNN_Block(10, 2, dilation_rates[0])
        self.DC_block2 = DCCNN_Block(20, 2, dilation_rates[1])
        self.DC_block3 = DCCNN_Block(30, 2, dilation_rates[2])
        self.DC_block4 = DCCNN_Block(40, 2, dilation_rates[3])
        self.dropout = nn.Dropout(0.1)

        self.RNN = nn.RNN(input_size=20, hidden_size=2,
                          num_layers=2, nonlinearity='tanh', batch_first=True)
        self.out1 = nn.Linear(in_features=2, out_features=18, bias=True)
        self.out2 = nn.Linear(in_features =18, out_features =1, bias=True)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        laa = self.laa(x)
        l1a, l1b = self.DC_block1(laa)
        l2a, l2b = self.DC_block2(l1a)
        l3a, l3b = self.DC_block3(l2a)
        l4a, l4b = self.DC_block4(l3a)
        l7b = self.dropout(l4b)
        l8 = laa+l1b+l2b+l3b+l4b+l7b
        l9 = self.RNN(l8)
        l10 = torch.flatten(l9[0], start_dim=1)
        l11 = self.tanh(self.out1(l10))
        l12 = self.out2(l11)
        
        return l12


torch.cuda.empty_cache()
mynetwork = Model_Build_SN()
batch_size = 100
print('Model inputs')
print(summary2(mynetwork.cpu(), torch.zeros((100, 1, 20)), show_input=True))
print('Model Outputs')
print(summary2(mynetwork.cpu(), torch.zeros((100, 1, 20)), show_input=False))

mynetwork = Model_Build_SN().to(device)

#%% Loss and optimizer
class custom_loss(nn.Module):
    def __init__(self, weight=None, size_average=False):
        super(custom_loss, self).__init__()
        #MSE = nn.MSELoss() 
        
    def forward(self, output, targets):
        
        #F_01 = 0.22360679774*torch.sqrt(torch.tensor((math.pi)))*torch.erf(2.2360679774*inputs)
        F_05 = 0.5*1.772453*torch.erf(output)
        output2 = torch.sum(torch.matmul(F_05,torch.transpose(h_s_tns, 0,1)),1)
        
        return torch.mean(torch.pow(targets[:,0]-output2[:], 2))
    
loss_fn = custom_loss().to(device)
#MSE = nn.MSELoss().to(device) 
criterion = loss_fn
lr=0.002;
optimizer = torch.optim.AdamW(mynetwork.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.01, amsgrad=False)
num_epochs = 50
#%%
train = torch.utils.data.TensorDataset(X_noise_train_tns.to(device), Y_noise_train_tns.to(device))
test = torch.utils.data.TensorDataset(X_noise_test_tns.to(device), Y_noise_test_tns.to(device))

batch_size = 512

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

n_iterations = math.ceil(len(X_noise_train)/batch_size)
print(n_iterations)
validation_iteration = math.ceil(len(X_noise_test)/batch_size)

PATH = 'path to save the model during training'

epoch_loss_saved = []
valid_loss_saved = []
since = time.time()
for epoch in range(num_epochs):
    train_loss = 0.0
    mynetwork.train()
    for step, (X, Y) in enumerate(train_loader):
    
        optimizer.zero_grad()
        outputs = mynetwork(X.to(device))
        loss = criterion(outputs, Y.to(device))

        # Backward and optimize
        loss.backward()
        optimizer.step()
        train_loss += loss.item() #
        
        if (step+1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{n_iterations}], Loss: {train_loss/step:.5f}')
            print('/n', outputs[10,0],Y[10,0])
        
    mynetwork.eval() 
    valid_loss = 0.0
    with torch.no_grad():
        for step, (X_val, Y_val) in enumerate(test_loader):
       
          # Forward Pass
          pred_val = mynetwork(X_val.to(device))
          # Find the Loss
          loss = criterion(pred_val,Y_val.to(device))
          # Calculate Loss
          valid_loss += loss.item()
          
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / n_iterations} \t\t Validation Loss: {valid_loss / validation_iteration}')
    epoch_loss_saved.append(train_loss / n_iterations)
    valid_loss_saved.append(valid_loss / validation_iteration)

    if valid_loss_saved[epoch] <= np.min(valid_loss_saved):
        torch.save(mynetwork, PATH)
        print('model saved')
    
    if (epoch+1)%3==0 and sum(valid_loss_saved[epoch-2:epoch+1] > np.min(valid_loss_saved))==3:
        lr = lr/2
        optimizer.param_groups[0]['lr'] = lr
        print("learning rate = ", optimizer.param_groups[0]['lr'])
 
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

#%%  saving model
PATH = 'path to save final model'
torch.save(mynetwork, PATH)

#%% loading the best model
PATH = 'path to load model'
mynetwork = torch.load(PATH)

#%% plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(epoch_loss_saved, "-b", label="Training Loss")
plt.plot(valid_loss_saved, "-r", label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(loc="upper right",fontsize='large')
plt.show()

#%% save the results 
DataName= 'DNoiseNet_FB_MLP_BGH_train_loss_0.5_15.csv'
np.savetxt("C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/GPU_2022_6_6/Previous paper/pytorch/saved_result2/"+DataName, epoch_loss_saved, delimiter=",")

DataName= 'DNoiseNet_FB_MLP_BGH_valid_loss_0.5_15.csv'
np.savetxt("C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/GPU_2022_6_6/Previous paper/pytorch/saved_result2/"+DataName, valid_loss_saved, delimiter=",")

#%% calculate the noise attenuation metric

X_test = X_noise_test_tns.to(device)

mynetwork.eval() 
since = time.time()
with torch.no_grad():
      pred_test = mynetwork(X_test)
time_elapsed = time.time() - since
#print(f'7.5 second of data : {time_elapsed} s' )

pred_test1 = 0.5*1.772453*torch.erf(pred_test)
pred_test2_1 = torch.sum(torch.matmul(pred_test1,torch.transpose(h_s_tns, 0,1)),1)
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

#%% MLP

class custom_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(custom_loss, self).__init__()

    def forward(self, output, targets):
        
        F_05 = 0.5*1.772453*torch.erf(output)
        output2 = torch.sum(h_s_tns*F_05)
        return targets-output2

loss_fn = custom_loss().to(device)
criterion = loss_fn

#%% Testing on feedback using MLP
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
DataName= 'DNoiseNet_FB_MLP_BGH_test_0.5.csv'
np.savetxt("path to save the results"+DataName, Y_noise_test, delimiter=",")

DataName= 'DNoiseNet_FB_MLP_BGH_pred_test_0.5.csv'
np.savetxt("path to save the results"+DataName, pred_test1, delimiter=",")

#%% calculate inference time of network developed by pytorch
device = torch.device("cuda")
mynetwork.to(device)
dummy_input = torch.randn(1, 1,20, dtype=torch.float32).to(device)
# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = mynetwork(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        dummy_input.to(device)
        _ = mynetwork(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        dummy_input.cpu()
        timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)

print('average inference time [ms]: ',mean_syn)
print('number of samples delay: ',16000*mean_syn/1000)

#%% calculate inference time of network developed by pytorch
# this time, we will used float16 for model and input

device = torch.device("cuda")
half_network = mynetwork.half().to(device)
dummy_input = torch.randn(1, 1,20, dtype=torch.float16).to(device)
# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = half_network(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = half_network(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)

print('average inference time [ms]: ',mean_syn)
print('number of samples delay: ',16000*mean_syn/1000)


#%% calculate the inference time just for transfering data

device = torch.device("cuda")
mynetwork.to(device)
dummy_input = torch.randn(20, dtype=torch.float32).to(device)
# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 1000
timings=np.zeros((repetitions,1))

# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        dummy_input.to(device)*10
        ender.record()
        # WAIT FOR GPU SYNC
        dummy_input.cpu()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)

print('Startup time [micro second]:',1000*mean_syn)
print('Maximum anti-noise generation speed of DNoiseNet by just considering communication time [kHz]= ', 1/mean_syn)


