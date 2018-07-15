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
def makelistofhits(hit_id, lengthofinfo):
    cell_rows = cells['hit_id'] == hit_id
    cell_hits = cells[cell_rows].drop('hit_id',axis=1).sort_values(by=['value'],ascending=False)
#    total = cell_hits["value"].sum()
    total = 1.0
    hit_row = hits['hit_id'] == hit_id
    volume_id = hits[hit_row]['volume_id'].item()
    layer_id = hits[hit_row]['layer_id'].item()
    module_id = hits[hit_row]['module_id'].item()
    detector._load_element_info(volid,layer_id,module_id)
    nCellsU = int(2*detector.module_maxhu/detector.pitch_u)
    nCellsV = int(2*detector.module_hv/detector.pitch_v)
    temp = (np.array(cell_hits)[:,0:3]*[1./nCellsU,1./nCellsV,1./total]).flatten()
    temp = temp[:min(len(temp), lengthofinfo)]
    return np.pad(temp,(0,lengthofinfo-len(temp)),'constant', constant_values=0)
    
    
    
    
    