import numpy as np
import pandas as pd
from trackml.dataset import load_event

def top_k_accuracy_score(y_true, y_pred, k=5, normalize=True, weights = None, argsorted = None):

    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)

    num_obs, num_labels = y_pred.shape
    idx = num_labels - k - 1
    counter = 0
    if argsorted is None:
        argsorted = np.argsort(y_pred, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx+1:]:
            if weights is None:
                counter += 1
            else:
                counter += weights[i]
    if normalize:
        if weights is None:
            return 1.* counter / num_obs
        else:
            return 1.*counter / np.sum(weights)
    else:
        return counter

    
def list_k_fails(y_true, y_pred, track_unique_ids, k=5, normalize=True, weights = None, argsorted = None):

    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)
        
    num_obs, num_labels = y_pred.shape
    idx = num_labels - k - 1
    counter = 0
    if argsorted is None:
        argsorted = np.argsort(y_pred, axis=1)
        
    returnlist = []
    
    for i in range(num_obs):
        if y_true[i] not in argsorted[i, idx+1:]:
            returnlist.append(track_unique_ids[i])
            
    return np.array(returnlist)

