import numpy as np
import pandas as pd

class Transforms:
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
        Output: Local coordinates (u, v, w)
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

if __name__ == '__main__':
    location = '/media/isaacson/DataStorage/kaggle/competitions/trackml-particle-identification/detectors.csv'
    trans = Transforms(location) 
    print(trans.LocalToGlobal(275,10,14,2,1))
    x,y,z = trans.LocalToGlobal(275,10,14,2,1)
    print(trans.GlobalToLocal(x,y,z,14,2,1))
