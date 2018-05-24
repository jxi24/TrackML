import numpy as np
import numpy.random as rand
import pandas as pd

from trackml.dataset import load_event

from keras.preprocessing.sequence import pad_sequences

from sklearn import preprocessing

class VolumeHit:
    def __init__(self, event_file, std_scale = False):
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
        #coord_dtype = self.hits_xyzrphiphitheta[0].dtype
        if std_scale:
            coord_rescaler = preprocessing.StandardScaler().fit(self.hits_xyzrphiphitheta)
            self.hits_xyzrphiphitheta = coord_rescaler.transform(self.hits_xyzrphiphitheta)

        self.onehot_volume = {
                7:  np.array([1,0,0,0,0,0,0,0,0,0]),
                8:  np.array([0,1,0,0,0,0,0,0,0,0]),
                9:  np.array([0,0,1,0,0,0,0,0,0,0]),
                12: np.array([0,0,0,1,0,0,0,0,0,0]),
                13: np.array([0,0,0,0,1,0,0,0,0,0]),
                14: np.array([0,0,0,0,0,1,0,0,0,0]),
                16: np.array([0,0,0,0,0,0,1,0,0,0]),
                17: np.array([0,0,0,0,0,0,0,1,0,0]),
                18: np.array([0,0,0,0,0,0,0,0,1,0]),
                0:  np.array([0,0,0,0,0,0,0,0,0,1])
                }
        
        self.hits_vol = self.hits.values[:,4]

        # Collect all ids except for id 0.
        self.track_unique_ids = np.unique(np.append([0],self.truth.values[:,1]))
        self.track_unique_ids = self.track_unique_ids[1:]
        #rand.shuffle(track_unique_ids)
        
        # Preprocess tracks
        self.tracks = {}
        self.track_hits = {}
        self.track_coords = {}
        self.track_hits_vols = {}
        self.track_vol_onehot = {}
        
        for track_unique_id in self.track_unique_ids:
            hit_rows = self.truth['particle_id'] == track_unique_id
            self.tracks[track_unique_id] = self.truth[hit_rows].drop('particle_id',axis=1)
            self.track_hits[track_unique_id] = self.tracks[track_unique_id]['hit_id'].values[:]
            self.track_coords[track_unique_id] = []
            self.track_hits_vols[track_unique_id] = []
            self.track_vol_onehot[track_unique_id] = []

            for hit_id in np.nditer(self.track_hits[track_unique_id]):
                # Append to track_coords array
                self.track_coords[track_unique_id].append(self.hits_xyzrphiphitheta[hit_id-1])
        
                # Find volumes of hits
                self.track_vol_onehot[track_unique_id].append(self.onehot_volume[self.hits_vol[hit_id-1]])
            
            # Pad track_coords 
#            self.track_coords[track_unique_id] = np.array(self.track_coords[track_unique_id])
            #self.track_coords[track_unique_id] = pad_sequences(self.track_coords[track_unique_id],padding='post',dtype=self.track_coords[track_unique_id][0].dtype)
        
            # Pad track_vol_onehot
            if len(self.track_hits[track_unique_id]) > 1:
                self.track_vol_onehot[track_unique_id] = self.track_vol_onehot[track_unique_id][1:]
                self.track_vol_onehot[track_unique_id].append(self.onehot_volume[0])
            else:
                self.track_vol_onehot[track_unique_id] = [self.onehot_volume[0]]
            #self.track_vol_onehot[track_unique_id] = pad_sequences(self.track_vol_onehot[track_unique_id],padding='post')
            self.track_coords[track_unique_id] = np.array(self.track_coords[track_unique_id])

        #self.track_coords[track_unique_id] = pad_sequences(self.track_coords,padding='post',dtype=coord_dtype)
        #self.track_vol_onehot = pad_sequences(self.track_vol_onehot,padding='post')
    
    def create_batch(self,batchsize = 100):
        tracks_list = np.random.choice(len(self.track_unique_ids),batchsize,replace=False)
       
        track_hits_list_coords = []
        hits_vol_onehot = []
        for track in tracks_list:
            track_id = self.track_unique_ids[track]
            track_hits_list_coords.append(self.track_coords[track_id])
            hits_vol_onehot.append(self.track_vol_onehot[track_id])

        track_hits_list_coords = pad_sequences(track_hits_list_coords,padding='post',dtype=track_hits_list_coords[0].dtype)
        hits_vol_onehot = pad_sequences(hits_vol_onehot,padding='post')

        return track_hits_list_coords, hits_vol_onehot

    def batch_generator_random(self, batchsize = 100):
        while True:
            yield self.create_batch(batchsize)

if __name__ == "__main__":
    vh = VolumeHit('/media/isaacson/DataStorage/kaggle/competitions/trackml-particle-identification/train_100_events/event000001000')

    print(vh.create_batch(5))
