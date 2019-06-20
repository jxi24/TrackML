import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class Detector:
    def __init__(self,detector_file):
        self.detector = pd.read_csv(detector_file)

    def _load_element_info(self,volume_id,layer_id,module_id):
        volume = self.detector['volume_id'] == volume_id
        layer = self.detector['layer_id'] == layer_id
        module = self.detector['module_id'] == module_id
        detector_element = self.detector[volume & layer & module]

        self.cshift = np.array([detector_element['cx'].item(),
                                detector_element['cy'].item(),
                                detector_element['cz'].item()])

        self.rotation_matrix = np.matrix([
            [detector_element['rot_xu'].item(),detector_element['rot_xv'].item(),detector_element['rot_xw'].item()],
            [detector_element['rot_yu'].item(),detector_element['rot_yv'].item(),detector_element['rot_yw'].item()],
            [detector_element['rot_zu'].item(),detector_element['rot_zv'].item(),detector_element['rot_zw'].item()]
            ])

        self.pitch_u = detector_element['pitch_u'].item()
        self.pitch_v = detector_element['pitch_v'].item()

        self.module_t = detector_element['module_t'].item()
        self.module_minhu = detector_element['module_minhu'].item()
        self.module_maxhu = detector_element['module_maxhu'].item()
        self.module_hv = detector_element['module_hv'].item()

    def _position(self,ch,pitch,h):
        return (ch + 0.5) * pitch - h 

    def _positon_inv(self,val,pitch,h):
        return (val + h)/pitch - 0.5

    def _calc_hu(self,v):
        return (self.module_minhu * (self.module_hv + v) 
                + self.module_maxhu * (self.module_hv - v)) / (2.0 * self.module_hv)

    def GlobalToLocal(self,x,y,z,volume_id,layer_id,module_id):
        """ 
        Purpose: Converts the position in global coordinates to the local coordinates.
        Input: x, y, z -> Global Position
               volume_id, layer_id, module_id -> identifications for detector lookup
        Output: Local coordinates (ch0, ch1)
        """

        self._load_element_info(volume_id,layer_id,module_id)
        u, v, w = np.array(np.transpose(self.rotation_matrix).dot(np.array([x,y,z]) - self.cshift)).flatten()

        return self._positon_inv(u,self.pitch_u,self._calc_hu(v)), \
               self._positon_inv(v,self.pitch_v,self.module_hv)
    
    def LocalToGlobal(self,ch0,ch1,volume_id,layer_id,module_id):
        """ 
        Purpose: Converts the position in local coordinates to the global coordinates.
        Input: ch0, ch1 -> channel location
               volume_id, layer_id, module_id -> volume, layer, and module identifications for transformation
        Output: Local coordinates (u, v, w)
        """
        self._load_element_info(volume_id,layer_id,module_id)
        v = self._position(ch1,self.pitch_v,self.module_hv)
        u = self._position(ch0,self.pitch_u,self._calc_hu(v))

        return np.array(self.rotation_matrix.dot(np.array([u,v,0.])) + self.cshift).flatten()
        
    def GlobalToLocalBatch(self,hits):
        """ 
        Purpose: Converts the position in global coordinates to the local coordinates.
        Input: DataBase containing:
                - x, y, z -> Global Position
                - volume_id, layer_id, module_id -> identifications for detector lookup
        Output: Local coordinates array (ch0, ch1)
        """

        channels = np.empty([0,2])
        grouped = hits.groupby(['volume_id', 'layer_id', 'module_id'], as_index=False)
        for name, group in grouped:
            # Load information about the group
            element = detector[(detector['volume_id'] == name[0]) & (detector['layer_id'] == name[1]) & (detector['module_id'] == name[2])]
            rot_mat = element[['rot_xu','rot_yu','rot_zu','rot_xv','rot_yv','rot_zv','rot_xw','rot_yw','rot_zw']].values.reshape(3,3)
            cshift = element[['cx','cy','cz']].values
            pitch_u, pitch_v = element[['pitch_u','pitch_v']].values.T
            self.module_minhu, self.module_maxhu, self.module_hv = element[['module_minhu','module_maxhu','module_hv']].values.T

            # Calculate ch0 and ch1 values
            x,y,z = group[['x','y','z']].values.T
            u,v,w = rot_mat.dot([x,y,z] - chsift.T)
            ch0 = self._position_inv(u,pitch_u,self._calc_hu(v))
            ch1 = self._position_inv(v,pitch_v,self.module_hv)
            channels = np.append(channels, np.array([ch0,ch1]).T, 0) 

        return channels
        
    def LocalToGlobalBatch(self,hits):
        """ 
        Purpose: Converts the position in local coordinates to the global coordinates.
        Input: DataBase containing:
                - ch0, ch1 -> Local Position
                - volume_id, layer_id, module_id -> identifications for detector lookup
        Output: Global coordinates array (x, y, z)
        """

        position = np.empty([0,3])
        grouped = hits.groupby(['volume_id', 'layer_id', 'module_id'], as_index=False)
        for name, group in grouped:
            # Load information about the group
            element = detector[(detector['volume_id'] == name[0]) & (detector['layer_id'] == name[1]) & (detector['module_id'] == name[2])]
            rot_mat = element[['rot_xu','rot_xv','rot_xw','rot_yu','rot_yv','rot_yw','rot_zu','rot_zv','rot_zw']].values.reshape(3,3)
            cshift = element[['cx','cy','cz']].values
            pitch_u, pitch_v = element[['pitch_u','pitch_v']].values.T
            self.module_minhu, self.module_maxhu, self.module_hv = element[['module_minhu','module_maxhu','module_hv']].values.T

            # Calculate ch0 and ch1 values
            ch0, ch1 = group[['ch0','ch1']].values.T
            v = self._position(ch1,pitch_v,self.module_hv)
            u = self._position(ch0,pitch_u,self._calc_hu(v))
            w = np.zeros_like(u)
            x,y,z = rot_mat.dot([u,v,w]) + chsift.T
            position = np.append(position, np.array([x,y,z]).T, 0) 

        return position
        
    def LocalToGlobalMom(self,u, v, w, volume_id,layer_id,module_id):
        """ 
        Purpose: Converts the position in local coordinates to the global coordinates.
        Input: pu, pv, pw -> momentum in local coordinates
               volume_id, layer_id, module_id -> volume, layer, and module identifications for transformation
        Output: Local coordinates (u, v, w)
        """
        self._load_element_info(volume_id,layer_id,module_id)

        return np.array(self.rotation_matrix.dot(np.array([u,v,w]))).flatten()   
        
        
    def GlobalToLocalMom(self,x,y,z,volume_id,layer_id,module_id):
        """ 
        Purpose: Converts the momentum in global coordinates to the local coordinates.
        Input: px, py, pz -> Global Momentum
               volume_id, layer_id, module_id -> identifications for detector lookup
        Output: Local coordinates (pu, pv, pw)
        """

        self._load_element_info(volume_id,layer_id,module_id)
        u, v, w = np.array(np.transpose(self.rotation_matrix).dot(np.array([x,y,z]))).flatten()
    
        return u, v, w

    def TransformGlobalToLocal(self,hits):
        element = self.detector[(self.detector['volume_id'] == hits.name[0]) & (self.detector['layer_id'] == hits.name[1]) & (self.detector['module_id'] == hits.name[2])]
        rot_mat = element[['rot_xu','rot_yu','rot_zu','rot_xv','rot_yv','rot_zv','rot_xw','rot_yw','rot_zw']].values.reshape(3,3)
        x,y,z = hits[['tpx','tpy','tpz']].values.T
        pu,pv,pw = rot_mat.dot([x,y,z])
        return pd.DataFrame({"hit_id": hits['hit_id'], "pu": pu, "pv": pv, "pw": pw})

    def TransformLocalToGlobal(self,hits):
        element = self.detector[(self.detector['volume_id'] == hits.name[0]) & (self.detector['layer_id'] == hits.name[1]) & (self.detector['module_id'] == hits.name[2])]
        rot_mat = element[['rot_xu','rot_xv','rot_xw','rot_yu','rot_yv','rot_yw','rot_zu','rot_zv','rot_zw']].values.reshape(3,3)
        u,v,w = hits[['pu','pv','pw']].values.T
        px,py,pz = rot_mat.dot([u,v,w])
        return pd.DataFrame({"hit_id": hits['hit_id'], "px": px, "py": py, "pz": pz})

    def GlobalToLocalMomBatch(self,hits):
        """
        Purpose: Converts the momentum in global coordinates to the local coordinates.
        Input: Hit database with information on px, py, pz, volume_id, layer_id, and module_id
        Output: Array of coordinates (pu, pv, pw)
        """

        return hits.groupby(['volume_id', 'layer_id', 'module_id']).apply(self.TransformGlobalToLocal) 


    def LocalToGobalMomBatch(self,hits):
        """
        Purpose: Converts the momentum in local coordinates to the global coordinates.
        Input: Hit database with information on pu, pv, pw, volume_id, layer_id, and module_id
        Output: Array of coordinates (px, py, pz)
        """

        return hits.groupby(['volume_id','layer_id','module_id']).apply(self.TransformLocalToGlobal)

    def GlobalToLocalMomBatchNorm(self, hits):
        """
        Purpose: Converts the momentum in global coordinates to local coordinates and normalizes them
        Input: Hit database with information on px, py, pz, volume_id, layer_id, and module_id
        Output: Array of coordinates normalized to 1 (pu,pv,pw)/|p|
        """

        momentum = self.GlobalToLocalMomBatch(hits)
        momentum[['pu','pv','pw']] = momentum[['pu','pv','pw']]/np.linalg.norm(momentum[['pu','pv','pw']],axis=1,keepdims=True)
        return momentum

    def HitsToImage(self, cell_hits, volume_id, layer_id, module_id):
        self._load_element_info(volume_id,layer_id,module_id)
        nCellsU = int(2*self.module_maxhu/self.pitch_u)
        nCellsV = int(2*self.module_hv/self.pitch_v)
    
        module_img = np.zeros((nCellsU,nCellsV))

        central_u = 0
        central_v = 0
        count = 0
        for index, row in cell_hits.iterrows():
            module_img[int(row['ch0']-1)][int(row['ch1']-1)] = row['value']
            central_u += int(row['ch0']-1)
            central_v += int(row['ch1']-1)
            count += 1.

        fig = plt.figure()
#        ax = fig.add_subplot(121)
        im = plt.imshow(module_img, interpolation='nearest', origin='low',
                extent=[0,nCellsU-1,0,nCellsV-1])

        return module_img

        center = (int(central_u/count),int(central_v/count))

        aspect_ratio = self.module_hv/self.module_maxhu

        nU = 50
        nV = int(50*aspect_ratio)
        centered_img = np.zeros((nU,nV))
        for i in xrange(nU):
            for j in xrange(nV):
                centered_img[i][j] = module_img[center[0]-nU/2+i][center[1]-nV/2+j]

#        ax = fig.add_subplot(122)
#        im = plt.imshow(centered_img, interpolation='nearest', origin='low',
#                extent=[0, nU, 0, nV])
#        plt.show()

        return module_img, centered_img

if __name__ == '__main__':
    from trackml.dataset import load_event
    
    hits, cells, particles, truth = load_event('../training_data/event000001000')

    location = '../training_data/detectors.csv'

    hit_id = 17667
    detector = Detector(location) 
    cell_rows = cells['hit_id'] == hit_id
    cell_hits = cells[cell_rows].drop('hit_id',axis=1)
    hit_row = hits['hit_id'] == hit_id
    volume_id = hits[hit_row]['volume_id'].item()
    layer_id = hits[hit_row]['layer_id'].item()
    module_id = hits[hit_row]['module_id'].item()

    full_img = detector.HitsToImage(cell_hits,volume_id,layer_id,module_id)

    plt.show()
