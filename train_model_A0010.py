from __future__ import print_function
import numpy as np
import numpy.random as rand
import pandas as pd
from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import time
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Masking
from keras.layers import SimpleRNN, LSTM, GRU, TimeDistributed
from keras import metrics
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.models import load_model
from itertools import cycle

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts
        
myargs = getopts(sys.argv)
if '-o' not in myargs:
    myargs['-o'] = '/data1/users/jcollins/myoutput'
    print("Setting default -o:", myargs['-o'])

class monitor_training(keras.callbacks.Callback):
    def __init__(self, epochs = 100, plot_period = 20, accs=['acc'], imagename = 'img'):
        self.epochs = epochs
        self.plot_period = plot_period
        self.losslist = []
        self.acclists = {}
        self.accs = accs
        self.imagename = imagename
        for acc in accs:
            self.acclists[acc] = []
                
    def on_train_begin(self, logs={}):
        pass
    
    def on_epoch_end(self, epoch, logs={}):
        self.losslist.append(logs['loss'])
        for acc in self.accs:
            self.acclists[acc].append(logs[acc])
            
        if (epoch+1)%self.plot_period == 0:
            plt.close('all')
            fig, ax1a = plt.subplots()
            ax1b = ax1a.twinx()
            linestyles = ['-','--','-.',':']
            linestyle_iter = cycle(linestyles)
            
            for acc in self.accs:
                ax1b.plot(np.arange(1, len(self.acclists[acc])+1),self.acclists[acc],color='C0',label=acc, linestyle=next(linestyle_iter))
                ax1a.plot(np.arange(1, len(self.acclists[acc])+1),self.losslist,color='C1',label='Loss')
                

            ax1a.set_ylim(0,10)
            ax1b.set_ylim(0,1)
            ax1a.set_xlabel('Epoch')
            ax1a.set_ylabel('Loss')
            ax1b.set_ylabel('Train Accuracy')
            ax1a.grid()
            plt.savefig(self.imagename + '.png')
            

batchsize = 100
epochs = 1000
from tools.batching_v5 import ModuleHit
event_files = ['/data1/users/jcollins/TrackML/data/event00000' + str(i) for i in range(1001,2001)]
vh = ModuleHit(event_files,'../TrackML_data/detectors.csv',std_scale = True,verbose = 1,batch_size=batchsize)

metrics=['acc']
metric_names=['acc']

my_monitor_training = monitor_training(epochs=epochs,plot_period=1,accs=metric_names, imagename = myargs['-o'] + '_lossaccplot')
adam_opt = keras.optimizers.Adam(lr=0.003, beta_1=0.998, beta_2=0.999, epsilon=None, decay=1e-4, amsgrad=False)

# define RNN
model = Sequential()
input_dim=vh.return_input_dim()
model.add(Masking(mask_value=0., input_shape=(None,input_dim)))
model.add(GRU(100, return_sequences=True))
model.add(GRU(500, return_sequences=True))
model.add(GRU(1000, return_sequences=True))
model.add(TimeDistributed(Dense(vh.num_modules+1, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer=adam_opt,
                            metrics=metrics)
model.save(myargs['-o'] + '_0.h5')
for epoch in range(epochs):
        model.fit_generator(vh,epochs=epoch+1,verbose=2,
                            steps_per_epoch=None,max_queue_size=100,use_multiprocessing=True, shuffle = False,
                            workers = 5, initial_epoch=epoch,
                            callbacks = [my_monitor_training])
        if (epoch+1)%10 == 0:
            model.save(myargs['-o'] + '_' + str(epoch+1) + '.h5')
        sys.stdout.flush()
