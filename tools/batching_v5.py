from __future__ import print_function
import numpy as np
import numpy.random as rand
import pandas as pd
from keras.utils import Sequence
from trackml.dataset import load_event
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
import time
import pickle

class ModuleHit(Sequence):
    def __init__(self, event_files, detector_file, std_scale = False, verbose = 0, batch_size = 100):
        self.std_scale = std_scale
        self.event_files = event_files
        self.event_iter = 0
        self.verbose = verbose
        self.init_encoding(detector_file)
        self.batch_size = batch_size
        self.track_unique_ids = {}
        self.init_event(event_files[0], first_init = True)

    def return_data(self):
        return self.track_coords, self.track_mod_onehot, self.track_unique_ids

    def return_input_dim(self):
        return len(self.hits_input[0])
    
    def __len__(self):
        return int(np.ceil(len(self.track_unique_ids) / float(self.batch_size)))
        
    def on_epoch_end(self):
        self.event_iter = self.event_iter + 1
        if self.event_iter >= len(self.event_files):
            self.event_iter = 0
            rand.shuffle(self.event_files)
        self.init_event(self.event_files[self.event_iter], first_init = False)
        
    def init_encoding(self, detector_file):
        detector = pd.read_csv(detector_file)
        self.unique_module_ids = detector.values[:,:3]
        num_modules = len(self.unique_module_ids)
        self.num_modules = num_modules
        mapped_arrays = np.identity(num_modules + 1)
        self.onehot_module = {}
        for i, module in enumerate(self.unique_module_ids):
            self.onehot_module[tuple(module)] = mapped_arrays[i]
        self.onehot_module[(0.,0.,0.)] = mapped_arrays[-1]
        self.onehot_module[(0,0,0)] = mapped_arrays[-1]

        unique_layer_ids = np.unique(detector.values[:,:2],axis=0)
        num_layers = len(unique_layer_ids)
        mapped_layer_arrays = np.identity(num_layers)
        self.onehot_layer = {}
        for i, layer in enumerate(unique_layer_ids):
            self.onehot_layer[tuple(layer)] = mapped_layer_arrays[i]
        
    def init_event(self, event_file, first_init = False):
        if self.verbose > 0:
            print('\n\nLoading event:', event_file)

        self.hits, self.cells, self.particles, self.truth = load_event(event_file)

        hits_xyz = self.hits.values[:,1:4]
        hits_r = np.sqrt(np.power(hits_xyz[:,0],2)+np.power(hits_xyz[:,1],2))

        hits_phi_x = np.sign(hits_xyz[:,1]) * np.arccos(hits_xyz[:,0] / hits_r)
        hits_phi_y = np.sign(-hits_xyz[:,0]) * np.arccos(hits_xyz[:,1] / hits_r)
        hits_theta = np.arctan2(hits_xyz[:,2], hits_r)

        hits_layers = self.hits.values[:,4:6]
        hits_layers_onehot = np.array([self.onehot_layer[tuple(hits_layer)] for hits_layer in hits_layers])
        
        hits_xyzrphiphitheta = np.concatenate((
            hits_xyz,
            np.reshape(hits_r,(-1,1)),
            np.reshape(hits_phi_x,(-1,1)),
            np.reshape(hits_phi_y,(-1,1)),
            np.reshape(hits_theta,(-1,1))),
            axis=1)

        if self.std_scale:
            if first_init:
                self.coord_rescaler = preprocessing.StandardScaler().fit(hits_xyzrphiphitheta)
            hits_xyzrphiphitheta = self.coord_rescaler.transform(hits_xyzrphiphitheta)

        self.hits_input = np.append(hits_xyzrphiphitheta,hits_layers_onehot,axis=1)
            
        hits_module_array = self.hits.values[:,4:]
        self.hits_module = []
        for module in hits_module_array:
            self.hits_module.append(tuple(module))
            
        # Collect all ids except for id 0.
        self.track_unique_ids = np.unique(np.append([0],self.truth.values[:,1]))
        self.track_unique_ids = self.track_unique_ids[1:]
        rand.shuffle(self.track_unique_ids)
        self.ntracks = len(self.track_unique_ids)

        self.track_hit_dict = pickle.load( open( event_file + "-trackdict.p", "rb" ) )
        
        if self.verbose > 0:
             print('Finished loading event:', event_file)
            
    def __getitem__(self, idx):

#        print(idx, self.__len__())
        
        tracks_id_list = self.track_unique_ids[idx * self.batch_size:(idx + 1) * self.batch_size]

        track_hits = {}
        track_hits_list_coords = []
        hits_mod_onehot = []
        for track_unique_id in tracks_id_list:
            track_hits = self.track_hit_dict[track_unique_id]
            track_hits_coords = []
            track_mod_onehot = []
            for hit_id in track_hits:
                track_hits_coords.append(self.hits_input[hit_id-1])
                track_mod_onehot.append(self.onehot_module[self.hits_module[hit_id-1]])
            if len(track_hits_coords) > 1:
                track_mod_onehot = track_mod_onehot[1:]
                track_mod_onehot.append(self.onehot_module[(0,0,0)])
            else:
                track_mod_onehot = [self.onehot_module[(0,0,0)]]
            track_hits_coords = np.array(track_hits_coords)
            
            hits_mod_onehot.append(track_mod_onehot)
            track_hits_list_coords.append(track_hits_coords)
                
        track_hits_list_coords = pad_sequences(track_hits_list_coords,padding='post',dtype=track_hits_list_coords[0].dtype)
        hits_mod_onehot = pad_sequences(hits_mod_onehot,padding='post')

        return track_hits_list_coords, hits_mod_onehot

    def get_entire_event_for_NN_eval(self):

        tracks_id_list = self.track_unique_ids
        
        track_hits_list_coords = []
        hits_mod_onehot = []
        weights = []
        track_lengths = []
        for track_unique_id in tracks_id_list:
            track_hits = self.track_hit_dict[track_unique_id]
            track_hits_coords = []
            track_mod_onehot = []
            weights.append(np.sum(self.truth.values[track_hits-1,-1]))
            for hit_id in track_hits:
                track_hits_coords.append(self.hits_input[hit_id-1])
                track_mod_onehot.append(self.onehot_module[self.hits_module[hit_id-1]])
            if len(track_hits_coords) > 1:
                track_mod_onehot = track_mod_onehot[1:]
                track_mod_onehot.append(self.onehot_module[(0,0,0)])
            else:
                track_mod_onehot = [self.onehot_module[(0,0,0)]]
            track_hits_coords = np.array(track_hits_coords)
            track_lengths.append(len(track_hits))
            
            hits_mod_onehot.append(track_mod_onehot)
            track_hits_list_coords.append(track_hits_coords)

        
            
        track_hits_list_coords = pad_sequences(track_hits_list_coords,padding='post',dtype=track_hits_list_coords[0].dtype)
        hits_mod_onehot = pad_sequences(hits_mod_onehot,padding='post')
        weights = np.array(weights)
        track_lengths = np.array(track_lengths)
        
        return track_hits_list_coords, hits_mod_onehot, track_lengths, weights

    def get_track_hit_NNinput_realxyz_coords_and_module(self, track_id):

        track_hits = self.track_hit_dict[track_id]
        track_hits_coords = []
        track_mod_onehot = []
        hits_xyz = []
        for hit_id in track_hits:
            hits_xyz.append(self.hits.loc[self.hits["hit_id"] == hit_id].values[0,1:4])
            track_hits_coords.append(self.hits_input[hit_id-1])
            track_mod_onehot.append(self.onehot_module[self.hits_module[hit_id-1]])

        track_hits_coords = np.array(track_hits_coords)
        track_mod_onehot = np.array(track_mod_onehot)

        return track_hits_coords, hits_xyz, track_mod_onehot

    def convert_moduleonehot_to_module_id(module_onehot):
        module_absnum = np.argmax(module_onehot)
        return self.unique_module_ids[module_absnum]
