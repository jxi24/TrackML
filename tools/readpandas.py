import pandas as pd

def Get_Position(truth,hit_id):
    truth_row = truth['hit_id'] == hit_id
    truth_x = truth[truth_row]['tx']
    truth_y = truth[truth_row]['tx']
    truth_z = truth[truth_row]['tx']
    return (truth_x, truth_y, truth_z)

def Get_Momentum(truth,hit_id):
    truth_row = truth['hit_id'] == hit_id
    truth_px = truth[truth_row]['tpx'].item()
    truth_py = truth[truth_row]['tpy'].item()
    truth_pz = truth[truth_row]['tpz'].item()
    return (truth_px,truth_py,truth_pz)


