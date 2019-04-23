from __future__ import print_function
import numpy as np
import pandas as pd
from trackml.dataset import load_event
import pickle

for event in range(1000,4000):
    event_file = "/data1/users/jcollins/TrackML/data/event00000" + str(event)
    hits, truth = load_event(event_file, parts=['hits','truth'])
    
    track_unique_ids = np.unique(np.append([0],truth.values[:,1]))
    
    track_hits = {}
    
    for track_unique_id in track_unique_ids:
        hit_rows = truth['particle_id'] == track_unique_id
        tracks = truth[hit_rows].drop('particle_id',axis=1)
        track_hits[track_unique_id] = tracks['hit_id'].values[:]
        hits_xyz = hits.values[track_hits[track_unique_id]-1,1:4]
        argsorted = np.argsort(hits_xyz[:,2])
        rmin = np.sqrt(np.power(hits_xyz[argsorted][0,0],2) + np.power(hits_xyz[argsorted][0,1],2))
        rmax = np.sqrt(np.power(hits_xyz[argsorted][-1,0],2) + np.power(hits_xyz[argsorted][-1,1],2))
        if abs(rmin > rmax):
            argsorted = np.flip(argsorted,axis=0)
        track_hits[track_unique_id] = track_hits[track_unique_id][argsorted]    
        
    pickle.dump(track_hits, open(event_file + "-trackdict.p", "wb"))
