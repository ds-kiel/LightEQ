 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 23:15:52 2020

@author: mostafamousavi
"""
from __future__ import print_function
import tensorflow as tk
import tensorflow.keras
from tensorflow.keras.layers import add, Reshape, Dense,Input, TimeDistributed, Dropout, Activation, LSTM,GRU, Conv2D, Bidirectional, BatchNormalization 
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os
np.seterr(divide='ignore', invalid='ignore')
import h5py
from obspy.signal.trigger import trigger_onset
import pathlib

np.warnings.filterwarnings('ignore')
from tensorflow.keras import datasets, layers, models,Sequential

###############################################################################
###################################################################  Generator

class DataGenerator(tk.keras.utils.Sequence):
    
    """ Keras generator with preprocessing 
    Args:
        list_IDsx: list of waveform names, str
        file_name: name of hdf file containing waveforms data, str
        dim: waveform lenght, int       
        batch_size: batch size, int
        n_channels: number of channels, int
        phase_window: number of samples (window) around each phase, int
        shuffle: shuffeling the list, boolean
        norm_mode: normalization type, str
        add_event_r: chance for randomly adding a second event into the waveform, float
        add_noise_r: chance for randomly adding Gaussian noise into the waveform, float
        drop_channe_r: chance for randomly dropping some of the channels, float
        scale_amplitude_r: chance for randomly amplifying the waveform amplitude, float
        pre_emphasis: if raw waveform needs to be pre emphesis,  boolean

    Returns:
        Batches of two dictionaries:
        {'input': X}: pre-processed waveform as input
        {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection,
            P, and S respectively.
    """   
    def __init__(self,
                 data,
                 data_label,
                 file_name, 
                 dim=(151, 41), 
                 batch_size=32, 
                 n_channels=3, 
                 target_length =768, 
                 shuffle=True, 
                 norm_mode = 'max',
                 add_event_r = None,
                 shift_event_r = None,
                 add_noise_r = None, 
                 scale_amplitude_r = None,
                 pre_emphasis = False):
        
        'Initialization'
        
        self.file_name = file_name           
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.target_length = target_length
        self.shuffle = shuffle
        
        self.norm_mode = norm_mode
        self.add_event_r = add_event_r 
        self.shift_event_r = shift_event_r
        self.add_noise_r = add_noise_r
        self.scale_amplitude_r = scale_amplitude_r
        self.pre_emphasis = pre_emphasis
        
        self.data=data
        self.data_label=data_label
        
        #print(data.shape, data_label.shape)
        
        self.list_IDs_len = len(self.data[:,0,0,0])
        print(self.list_IDs_len,'*****')
        
        self.on_epoch_end()

    def __len__(self):
        #print('*************_len_')
        'Denotes the number of batches per epoch'
        return int(np.floor(self.list_IDs_len / self.batch_size))

    def __getitem__(self, index):

        X=self.data[index*self.batch_size:(index+1)*self.batch_size,:,:,:]
        y=self.data_label[index*self.batch_size:(index+1)*self.batch_size,:,:]
        return X, y
        

    def on_epoch_end(self):
        #print('*************on_epoch_end')
        self.indexes = np.arange(self.list_IDs_len)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)      
        

        
    



def generate_arrays_from_file(file_list, step):
    n_loops = int(np.ceil(len(file_list) / step))
    b = 0
    while True:
        for i in range(n_loops):
            e = i*step + step 
            if e > len(file_list):
                e = len(file_list)
            chunck = file_list[b:e]
            b=e
            yield chunck





class DataGenerator_test(tk.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, file_name, batch_size=32, dim=(151, 41), n_channels=3, norm_mode = 'max'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.on_epoch_end()
        self.norm_mode = norm_mode
        self.file_name = file_name        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        #print('*************Generate one batch of data..test')        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X = self.__data_generation(list_IDs_temp)
        return X
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def _normalize(self, data, mode = 'max'):
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            data /= max_data    
        elif mode == 'std':        
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data

    def __data_generation(self, list_IDs_temp):
        'readint the waveforms' 
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        fl = h5py.File(self.file_name, 'r')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            if ID.split('_')[-1] == 'EV':
                dataset = fl.get('data/'+str(ID))
                data = np.array(dataset)                   

            elif ID.split('_')[-1] == 'NO':
                dataset = fl.get('data/'+str(ID))
                data = np.array(dataset)            

            data = self._normalize(data, self.norm_mode) 

            for ch in range(self.n_channels): 
                bpf = data[:, ch]                        
                f, t, Pxx = signal.stft(bpf, fs = 100, nperseg=80)
                Pxx = np.abs(Pxx)
                X[i, :, :, ch] = Pxx.T 

        return X.astype('float32')  
        
def normalize(param, data, mode = 'max'):
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        data /= max_data    
    elif mode == 'std':        
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data

def data_generation(param, list_IDs_temp):
    'readint the waveforms' 
    lengthT=len(list_IDs_temp)
    #print(lengthT,'***************')
    X = np.empty((lengthT, param['dim'][0], param['dim'][1], param['n_channels']))
    fl = h5py.File(param['file_name'], 'r')

    # Generate data
    for i, ID in enumerate(list_IDs_temp):

        if ID.split('_')[-1] == 'EV':
            dataset = fl.get('data/'+str(ID))
            data = np.array(dataset)                   

        elif ID.split('_')[-1] == 'NO':
            dataset = fl.get('data/'+str(ID))
            data = np.array(dataset)            

        data = normalize(param,data, param['norm_mode']) 

        for ch in range(param['n_channels']): 
            bpf = data[:, ch]                        
            f, t, Pxx = signal.stft(bpf, fs = 100, nperseg=80)
            Pxx = np.abs(Pxx)
            X[i, :, :, ch] = Pxx.T 

    return X.astype('float32')    
    
    
    


def detector(args, yh1):

    """
    return two dictionaries and one numpy array:
        
        matches --> {detection statr-time:[ detection end-time,
                                           detection probability,]}
                
    """               
             
    detection = trigger_onset(yh1, args.detection_threshold , args.detection_threshold/2)

    EVENTS = {}
    matches = {}
                       
    if len(detection) > 0:        
        for ev in range(len(detection)):                                 
            D_prob = np.mean(yh1[detection[ev][0]:detection[ev][1]])
            D_prob = np.round(D_prob, 3)                  
            EVENTS.update({ detection[ev][0] : [D_prob, detection[ev][1]]})            

    for ev in EVENTS:
        bg = ev
        ed = EVENTS[ev][1]
        
        if int(ed-bg) >= 1:               
            matches.update({ bg:[ed, EVENTS[ev][0]
                                                ] })                                               
    return matches


 
    
def output_writter_test(args, 
                        dataset, 
                        evi, 
                        output_writer, 
                        csvfile, 
                        matches 
                        ):
    
    numberOFdetections = len(matches)
    
    if numberOFdetections != 0: 
        D_prob =  matches[list(matches)[0]][1]
    else: 
        D_prob = None

    
    if evi.split('_')[-1] == 'EV':                                     
        network_code = dataset.attrs['network_code']
        source_id = dataset.attrs['source_id']
        source_distance_km = dataset.attrs['source_distance_km']  
        snr_db = np.mean(dataset.attrs['snr_db'])
        trace_name = dataset.attrs['trace_name'] 
        trace_category = dataset.attrs['trace_category']            
        trace_start_time = dataset.attrs['trace_start_time'] 
        source_magnitude = dataset.attrs['source_magnitude'] 
        p_arrival_sample = dataset.attrs['p_arrival_sample'] 
        p_status = dataset.attrs['p_status'] 
        p_weight = dataset.attrs['p_weight'] 
        s_arrival_sample = dataset.attrs['s_arrival_sample'] 
        s_status = dataset.attrs['s_status'] 
        s_weight = dataset.attrs['s_weight'] 
        receiver_type = dataset.attrs['receiver_type']  
                   
    elif evi.split('_')[-1] == 'NO':               
        network_code = dataset.attrs['network_code']
        source_id = None
        source_distance_km = None 
        snr_db = None
        trace_name = dataset.attrs['trace_name'] 
        trace_category = dataset.attrs['trace_category']            
        trace_start_time = None
        source_magnitude = None
        p_arrival_sample = None
        p_status = None
        p_weight = None
        s_arrival_sample = None
        s_status = None
        s_weight = None
        receiver_type = dataset.attrs['receiver_type'] 

    output_writer.writerow([network_code, 
                            source_id, 
                            source_distance_km, 
                            snr_db, 
                            trace_name, 
                            trace_category, 
                            trace_start_time, 
                            source_magnitude,
                            p_arrival_sample, 
                            p_status, 
                            p_weight, 
                            s_arrival_sample, 
                            s_status,
                            s_weight,
                            receiver_type,                
                            numberOFdetections,
                            D_prob
                            
                            ]) 
    
    csvfile.flush()             
    





def plotter(ts, 
            dataset, 
            evi,
            args,
            save_figs,
            yh1,
            matches
            ):

    try:
        spt = int(dataset.attrs['p_arrival_sample']);
    except Exception:     
        spt = None
                    
    try:
        sst = int(dataset.attrs['s_arrival_sample']);
    except Exception:     
        sst = None
    
    data = np.array(dataset)
    
    fig = plt.figure()
    ax = fig.add_subplot(411)         
    plt.plot(data[:, 0], 'k')
    plt.rcParams["figure.figsize"] = (8,5)
    legend_properties = {'weight':'bold'}  
    plt.title(str(evi))
    plt.tight_layout()
    ymin, ymax = ax.get_ylim() 
    pl = None
    sl = None       

    
    ax = fig.add_subplot(412)   
    plt.plot(data[:, 1] , 'k')
    plt.tight_layout()                
       

    ax = fig.add_subplot(413) 
    plt.plot(data[:, 2], 'k')   
    plt.tight_layout()                
                    
    ax = fig.add_subplot(414)
    plt.plot(yh1, 'g--', alpha = 0.5, linewidth=1.5, label='Detection')
    plt.tight_layout()       
    plt.ylim((-0.1, 1.1))
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties) 
                        
    fig.savefig(os.path.join(save_figs, str(evi.split('/')[-1])+'.png')) 
    fig.savefig(os.path.join(save_figs, str(evi.split('/')[-1])+'.pdf')) 

  
 
    

############################################################# model

def lr_schedule(epoch):
    """
    Learning rate is scheduled to be reduced after 40, 60, 80, 90 epochs.
    """
    lr = 1e-3
    if epoch > 60:
        lr *= 0.5e-3
    elif epoch > 40:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1

        
    print('Learning rate: ', lr)
    return lr






def block_BiLSTM(inpR, filters, rnn_depth):
    """
    Returns LSTM residual blocks
    """
    x = inpR
    for i in range(rnn_depth):
        x_rnn = Bidirectional(GRU(filters, return_sequences=True))(x)
        x_rnn = Dropout(0.7)(x_rnn)
        if i > 0 :
           x = add([x, x_rnn])
        else:
           x = x_rnn      
    return x
     

def block_CNN(filters, ker, inpC): 
    """
    Returns CNN residual blocks
    """
    layer_1 = BatchNormalization()(inpC) 
    act_1 = Activation('relu')(layer_1) 

    conv_1 = Conv2D(filters, (ker, ker), padding = 'same')(act_1) 
    
    layer_2 = BatchNormalization()(conv_1) 
    act_2 = Activation('relu')(layer_2) 
  
    conv_2 = Conv2D(filters, (ker, ker), padding = 'same')(act_2) 
    return(conv_2) 


def convblock(cfilters,ckernels,cstrides,cchoice,inp):
    e=inp
    for i in range(cchoice):
        e = Conv2D(cfilters[i], ckernels[i], strides =cstrides[i],  padding = 'same', activation='relu')(e)
        e= tk.keras.layers.add([block_CNN(cfilters[i], ckernels[i]-2, e), e])
        
    return(e)
    
def LSTMblock(lunits,ldrops,lchoice,inp):
    e=inp
    for i in range(lchoice):
        e = lstm1 = LSTM(lunits[i], return_sequences=True)(e) 
        e = Dropout(ldrops[i])(e)
        e = BatchNormalization()(e)
        
    return(e)

def model_lighteq_model2( filtr1,filtr2,filtr3,filtr4,filtr5,str1,str2,str3,str4,str5,conv_size1,conv_size2,conv_size3,conv_size4,conv_size5,dropout_rate_cnn,dropout_rate_lstm1,dropout_rate_lstm2,dropout_rate_lstm3, dropout_rate_dense,lstm_unit1,lstm_unit2,lstm_unit3,cchoice,lchoice):
       
        
    cfilters=[filtr1,filtr2,filtr3,filtr4,filtr5]
    ckernels=[conv_size1,conv_size2,conv_size3,conv_size4,conv_size5]
    cstrides=[str1,str2,str3,str4,str5]
    lunits=[lstm_unit1,lstm_unit2,lstm_unit3]
    ldrops=[dropout_rate_lstm1,dropout_rate_lstm2,dropout_rate_lstm3]

    #model for rzjKo & Zp00o
    inp = Input(shape=(151,41,3), name='input')
    e = Conv2D(cfilters[0], ckernels[0], strides =cstrides[0],  padding = 'same', activation='relu')(inp)
    e= tk.keras.layers.add([block_CNN(cfilters[0], ckernels[0]-2, e), e])
    
    e = Conv2D(cfilters[1], ckernels[1], strides =cstrides[1],  padding = 'same', activation='relu')(e)
    e= tk.keras.layers.add([block_CNN(cfilters[1], ckernels[1]-2, e), e])
    
    e = Conv2D(cfilters[2], ckernels[2], strides =cstrides[2],  padding = 'same', activation='relu')(e)
    e= tk.keras.layers.add([block_CNN(cfilters[2], ckernels[2]-2, e), e])
    
    x = Dropout(dropout_rate_cnn)(e)

    shape = tk.keras.backend.int_shape(x)
    x = Reshape((shape[1], shape[2]*shape[3]))(x)
    
    e = lstm1 = LSTM(lunits[0], return_sequences=True, unroll=True)(x) 
    e = Dropout(ldrops[0])(e)
    e = BatchNormalization()(e)
    

    x = TimeDistributed(Dense(64, kernel_regularizer=l1(0.01), activation='relu'))(e)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate_dense)(x)
    x = TimeDistributed(Dense(1, kernel_regularizer=l1(0.01), activation='sigmoid'))(x)


    out_model = Model(inputs=inp, outputs=x)
    
    return out_model



def model_lighteq_model1( filtr1,filtr2,filtr3,filtr4,filtr5,str1,str2,str3,str4,str5,conv_size1,conv_size2,conv_size3,conv_size4,conv_size5,dropout_rate_cnn,dropout_rate_lstm1,dropout_rate_lstm2,dropout_rate_lstm3, dropout_rate_dense,lstm_unit1,lstm_unit2,lstm_unit3,cchoice,lchoice):
       
        
    cfilters=[filtr1,filtr2,filtr3,filtr4,filtr5]
    ckernels=[conv_size1,conv_size2,conv_size3,conv_size4,conv_size5]
    cstrides=[str1,str2,str3,str4,str5]
    lunits=[lstm_unit1,lstm_unit2,lstm_unit3]
    ldrops=[dropout_rate_lstm1,dropout_rate_lstm2,dropout_rate_lstm3]

    
    
    #MODEL1
    inp = Input(shape=(151,41,3), name='input')
    e = Conv2D(cfilters[0], ckernels[0], strides =cstrides[0],  padding = 'same', activation='relu')(inp)
    e= tk.keras.layers.add([block_CNN(cfilters[0], ckernels[0]-2, e), e])
    
    e = Conv2D(cfilters[1], ckernels[1], strides =cstrides[1],  padding = 'same', activation='relu')(e)
    e= tk.keras.layers.add([block_CNN(cfilters[1], ckernels[1]-2, e), e])
    
    e = Conv2D(cfilters[2], ckernels[2], strides =cstrides[2],  padding = 'same', activation='relu')(e)
    e= tk.keras.layers.add([block_CNN(cfilters[2], ckernels[2]-2, e), e])
    
    e = Conv2D(cfilters[3], ckernels[3], strides =cstrides[3],  padding = 'same', activation='relu')(e)
    e= tk.keras.layers.add([block_CNN(cfilters[3], ckernels[3]-2, e), e])
    
    e = Conv2D(cfilters[4], ckernels[4], strides =cstrides[4],  padding = 'same', activation='relu')(e)
    e= tk.keras.layers.add([block_CNN(cfilters[4], ckernels[4]-2, e), e])
    
    x = Dropout(dropout_rate_cnn)(e)

    shape = tk.keras.backend.int_shape(x)
    x = Reshape((shape[1], shape[2]*shape[3]))(x)
    
    e = lstm1 = LSTM(lunits[0], return_sequences=True, unroll=True)(x) 
    e = Dropout(ldrops[0])(e)
    e = BatchNormalization()(e)
    

    x = TimeDistributed(Dense(64, kernel_regularizer=l1(0.01), activation='relu'))(e)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate_dense)(x)
    x = TimeDistributed(Dense(1, kernel_regularizer=l1(0.01), activation='sigmoid'))(x)


    out_model = Model(inputs=inp, outputs=x)
    
    return out_model
    
   
    
    
def model_lighteq_model0( filtr1,filtr2,filtr3,filtr4,filtr5,str1,str2,str3,str4,str5,conv_size1,conv_size2,conv_size3,conv_size4,conv_size5,dropout_rate_cnn,dropout_rate_lstm1,dropout_rate_lstm2,dropout_rate_lstm3, dropout_rate_dense,lstm_unit1,lstm_unit2,lstm_unit3,cchoice,lchoice):
       
        
    cfilters=[filtr1,filtr2,filtr3,filtr4,filtr5]
    ckernels=[conv_size1,conv_size2,conv_size3,conv_size4,conv_size5]
    cstrides=[str1,str2,str3,str4,str5]
    lunits=[lstm_unit1,lstm_unit2,lstm_unit3]
    ldrops=[dropout_rate_lstm1,dropout_rate_lstm2,dropout_rate_lstm3]

    #Use exactly as it is for model0
    inp = Input(shape=(151,41,3), name='input')
    e = Conv2D(cfilters[0], ckernels[0], strides =2,  padding = 'same', activation='relu')(inp)
    e = BatchNormalization()(e) 
    e = Activation('relu')(e)
    
    act_1 = Dropout(dropout_rate_cnn)(e)

    shape = tk.keras.backend.int_shape(act_1)
    reshaped = Reshape((shape[1], shape[2]*shape[3]))(act_1)


    UNIlstm = LSTM(32, return_sequences=True, unroll=True)(reshaped)
    UNIlstm = Dropout(ldrops[0])(UNIlstm)  
    UNIlstm = BatchNormalization()(UNIlstm)

    
    dense_2 = TimeDistributed(Dense(64, kernel_regularizer=l1(0.01), activation='relu'))(UNIlstm)
    dense_2 = BatchNormalization()(dense_2)
    dense_2 = Dropout(dropout_rate_dense)(dense_2)

    dense_3 = TimeDistributed(Dense(1, kernel_regularizer=l1(0.01), activation='sigmoid'))(dense_2)

    out_model = Model(inputs=inp, outputs=dense_3)
    return out_model