import trackml
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from tools.detector import Detector
from tools.readpandas import Get_Momentum

from trackml.dataset import load_event, load_dataset
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.models import load_model
import sklearn.metrics as mt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


detector = Detector('/Users/pjfox/Dropbox/NN/TrackML/detectors.csv')
#convert truth momentum to u, v, w coordinates
def truthmom_to_uvw(a): # input is 'hit_id','tpx','tpy','tpz', 'volume_id', 'layer_id', 'module_id'
    b = detector.GlobalToLocalMom(a[1], a[2], a[3], a[4], a[5], a[6])
    norm = np.linalg.norm(b)
    return [b[0], b[1], b[2]]/norm

#go thru event and extract good hits in a particular volume (with volume_id == volid)    
def goodhit_and_directions(hits, truth, volid):
    df=hits.merge(truth)
    badhits=df.loc[df['particle_id'] == 0]
    goodhits=df.loc[df['particle_id'] != 0]
    selectedhits = np.array(goodhits.loc[(goodhits['volume_id'] == volid)][['hit_id','tpx','tpy','tpz', 'volume_id', 'layer_id', 'module_id']])
    directions = np.apply_along_axis(truthmom_to_uvw, 1, selectedhits)
    return selectedhits, directions
    
#make a list of all the cell info, pads with empty information or cuts it off after lengthofinfo/3 entries, ready for a NN to process, can be slow....
def makelistofhits(cells, hits, hit_id, lengthofinfo):
    cell_rows = cells['hit_id'] == hit_id
    cell_hits = cells[cell_rows].drop('hit_id', axis=1).sort_values(by=['value'], ascending=False)
#    total = cell_hits["value"].sum()
    total = 1.0
    hit_row = hits['hit_id'] == hit_id
    volume_id = hits[hit_row]['volume_id'].item()
    layer_id = hits[hit_row]['layer_id'].item()
    module_id = hits[hit_row]['module_id'].item()
    detector._load_element_info(volume_id,layer_id,module_id)
    nCellsU = int(2*detector.module_maxhu/detector.pitch_u)
    nCellsV = int(2*detector.module_hv/detector.pitch_v)
    temp = (np.array(cell_hits)[:,0:3]*[1./nCellsU,1./nCellsV,1./total]).flatten()
    temp = temp[:min(len(temp), lengthofinfo)]
    return np.pad(temp,(0,lengthofinfo-len(temp)),'constant', constant_values=0)

#  analyze whole event file
def analyse_event(event_number, volume_id, size_of_hit):
    hits, cells, particles, truth = load_event('/Users/pjfox/Dropbox/NN/TrackML/train_100_events/event00000' + str(event_number))
    selectedhits, directions = goodhit_and_directions(hits, truth, volume_id)
    hitinfo=np.empty([len(selectedhits), size_of_hit])
    for ii, hit_id in enumerate(selectedhits[:,0]):
        hitinfo[ii] = makelistofhits(cells, hits, hit_id, size_of_hit)
    np.save("hits_info_Vol_" + str(volume_id) + "_event_" + str(event_number) + ".npy", hitinfo)
    np.save("direction_info_Vol_" + str(volume_id) + "_event_" + str(event_number) + ".npy", directions)
    return None

#show distribution of cell charges (2d) and predicted direction compared to real direction in 3D
def display_hit(event_number, volume_id, hit_id, NN):
    hit = np.load("hits_info_Vol_" + str(volume_id) + "_event_" + str(event_number) + ".npy")[hit_id]
    truedirection = np.load("direction_info_Vol_" + str(volume_id) + "_event_" + str(event_number) + ".npy")[hit_id]
    predicted_direction = predict_direction(hit, NN)
    hit = hit[~(hit == 0)]
    hit = np.reshape(hit, (-1,3))
    ss = 50 * hit[:,2]
    veclength = max([np.sqrt((hit[:,0].max() - hit[:,0].min())**2 + (hit[:,1].max() - hit[:,1].min())**2),0.01])
    print veclength
    plt.scatter(hit[:,0], hit[:,1], s=ss)
    fig = plt.figure(figsize = (20,10))
    ax2 = fig.gca(projection='3d')
    ax2.quiver(hit[:,0].mean(), hit[:,1].mean(), 0, truedirection[0], truedirection[1], truedirection[2], length=veclength, normalize=True, color='g')
    ax2.quiver(hit[:,0].mean(), hit[:,1].mean(), 0, predicted_direction[0], predicted_direction[1], predicted_direction[2], length=veclength, normalize=True, color='r')
    ax2.scatter(hit[:,0], hit[:,1], np.zeros(len(hit)), s=ss)
    return None

# predict the direction from a hit, using neural net NN    
def predict_direction(hit, NN):
    predicteddirection = NN.predict(np.reshape(hit,(1,len(hit))),verbose=0)[0]
    predicteddirection = predicteddirection/np.linalg.norm(predicteddirection)
    return predicteddirection

# convert cartesian to spherical coordinates    
def Cart_to_Sph(vec):
    norm = np.linalg.norm(vec)
    theta = np.arccos(vec[2]/norm)
    phi = np.arctan2(vec[1],vec[0])
    return np.array([norm, theta, phi])
# convert spherical to cartesian
def Sph_to_Cart(vec):
    x = vec[0] * np.sin(vec[1]) * np.cos(vec[2])
    y = vec[0] * np.sin(vec[1]) * np.sin(vec[2])
    z = vec[0] * np.cos(vec[1])
    return np.array([x, y, z])