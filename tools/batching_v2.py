import numpy as np
import numpy.random as rand
import pandas as pd

from trackml.dataset import load_event

from keras.preprocessing.sequence import pad_sequences

from sklearn import preprocessing

class ModuleHit:
    def __init__(self, event_files, detector_file, std_scale = False, verbose = 0):
        self.std_scale = std_scale
        self.event_files = event_files
        self.event_iter = 0
        self.verbose = verbose
        self.init_encoding(detector_file)
        self.init_event(event_files[0], first_init = True)

    def init_encoding(self, detector_file):
        detector = pd.read_csv(detector_file)
        unique_module_ids = detector.values[:,:3]
        num_modules = len(unique_module_ids)
        self.num_modules = num_modules
        mapped_arrays = np.identity(num_modules + 1)
        self.onehot_module = {}
        for i, module in enumerate(unique_module_ids):
            self.onehot_module[tuple(module)] = mapped_arrays[i]
        self.onehot_module[(0.,0.,0.)] = mapped_arrays[-1]
        self.onehot_module[(0,0,0)] = mapped_arrays[-1]

    def init_event(self, event_file, first_init = False):
        if self.verbose > 0:
            print('\n\nLoading event:', event_file, '\n\n')

        self.hits, self.cells, self.particles, self.truth = load_event(event_file)

        self.hits_xyz = self.hits.values[:,1:4]
        self.hits_r = np.sqrt(np.power(self.hits_xyz[:,0],2)+np.power(self.hits_xyz[:,1],2))

        self.hits_phi_x = np.sign(self.hits_xyz[:,1]) * np.arccos(self.hits_xyz[:,0] / self.hits_r)
        self.hits_phi_y = np.sign(-self.hits_xyz[:,0]) * np.arccos(self.hits_xyz[:,1] / self.hits_r)
        self.hits_theta = np.arctan2(self.hits_xyz[:,2], self.hits_r)
        
        self.hits_xyzrphiphitheta = np.concatenate((
            self.hits_xyz,
            np.reshape(self.hits_r,(-1,1)),
            np.reshape(self.hits_phi_x,(-1,1)),
            np.reshape(self.hits_phi_y,(-1,1)),
            np.reshape(self.hits_theta,(-1,1))),
            axis=1)

        if self.std_scale:
            if first_init:
                self.coord_rescaler = preprocessing.StandardScaler().fit(self.hits_xyzrphiphitheta)
            self.hits_xyzrphiphitheta = self.coord_rescaler.transform(self.hits_xyzrphiphitheta)

        hits_module_array = self.hits.values[:,4:]
        self.hits_module = []
        for module in hits_module_array:
            self.hits_module.append(tuple(module))

        # Collect all ids except for id 0.
        self.track_unique_ids = np.unique(np.append([0],self.truth.values[:,1]))
        self.track_unique_ids = self.track_unique_ids[1:]
        rand.shuffle(self.track_unique_ids)
        self.ntracks = len(self.track_unique_ids)
        self.ntrack_iter = 0
        
        # Preprocess tracks
        self.tracks = {}
        self.track_hits = {}
        self.track_coords = {}
        self.track_hits_vols = {}
        self.track_mod_onehot = {}
        
        for track_unique_id in self.track_unique_ids:
            hit_rows = self.truth['particle_id'] == track_unique_id
            self.tracks[track_unique_id] = self.truth[hit_rows].drop('particle_id',axis=1)
            self.track_hits[track_unique_id] = self.tracks[track_unique_id]['hit_id'].values[:]
            self.track_coords[track_unique_id] = []
            self.track_hits_vols[track_unique_id] = []
            self.track_mod_onehot[track_unique_id] = []

            for hit_id in np.nditer(self.track_hits[track_unique_id]):
                # Append to track_coords array
                self.track_coords[track_unique_id].append(self.hits_xyzrphiphitheta[hit_id-1])
        
                # Find volumes of hits
                self.track_mod_onehot[track_unique_id].append(self.onehot_module[self.hits_module[hit_id-1]])
            
            # Pad track_vol_onehot
            if len(self.track_hits[track_unique_id]) > 1:
                self.track_mod_onehot[track_unique_id] = self.track_mod_onehot[track_unique_id][1:]
                self.track_mod_onehot[track_unique_id].append(self.onehot_module[(0,0,0)])
            else:
                self.track_mod_onehot[track_unique_id] = [self.onehot_module[(0,0,0)]]
            self.track_coords[track_unique_id] = np.array(self.track_coords[track_unique_id])

    def create_batch_from_tracks_list(self, tracks_list):
        track_hits_list_coords = []
        hits_mod_onehot = []
        for track in tracks_list:
            track_id = self.track_unique_ids[track]
            track_hits_list_coords.append(self.track_coords[track_id])
            hits_mod_onehot.append(self.track_mod_onehot[track_id])

        track_hits_list_coords = pad_sequences(track_hits_list_coords,padding='post',dtype=track_hits_list_coords[0].dtype)
        hits_mod_onehot = pad_sequences(hits_mod_onehot,padding='post')

        return track_hits_list_coords, hits_mod_onehot

    def create_random_batch(self,batchsize = 100):
        tracks_list = np.random.choice(len(self.track_unique_ids),batchsize,replace=False)
        return self.create_batch_from_tracks_list(tracks_list)

    def batch_generator_random(self, batchsize = 100):
        while True:
            yield self.create_random_batch(batchsize)

    def create_next_batch(self, batchsize = 100):
        #Check if we have reached the end of an event. If so, load the next event.
        if self.ntrack_iter == self.ntracks:
            self.init_next_event()
        if self.ntrack_iter + batchsize > self.ntracks:
            batchsize = self.ntracks - self.ntrack_iter
        
        tracks_list = np.arange(self.ntrack_iter, self.ntrack_iter + batchsize)
        self.ntrack_iter = self.ntrack_iter + batchsize
        return self.create_batch_from_tracks_list(tracks_list)

    def batch_generator_insequence(self, batchsize = 100):
        while True:
            yield self.create_next_batch(batchsize)

    def init_next_event(self):
        self.event_iter = self.event_iter + 1
        if self.event_iter >= len(self.event_files):
            self.event_iter = 0
            np.shuffle(self.event_files)
        self.init_event(self.event_files[self.event_iter], first_init = False)
