import numpy as np
import yt, caesar, h5py
import sys
from tqdm.auto import tqdm
from multiprocessing import Pool
#galaxy =

yt.set_log_level(50)

def _gassmoothedmasses(field, data):
    return data.ds.parameters['octree'][('PartType0', 'Masses')] 


gmass_sm = []
gmass_p = []
obj = caesar.load('/orange/narayanan/desika.narayanan/gizmo_runs/simba/m25n512/output/Groups/caesar_0305_z0.000.hdf5')

def open_snaps(galaxy):

    path = f'/orange/narayanan/s.lower/simba/desika_filtered_snaps/snap305/'
    fname = path+f'/galaxy_{galaxy}.hdf5'
    #print('------ loading snapshot in yt --------')                                                                                           
    ds = yt.load(fname)
#    data = ds.all_data()


    #print('------ loading caesar data ------')                                                                                                
#    obj = caesar.load('/orange/narayanan/desika.narayanan/gizmo_runs/simba/m25n512/output/Groups/caesar_0305_z0.000.hdf5')
    com = obj.galaxies[galaxy].pos.in_units('code_length')
    center = com
    box_len = ds.quan(100, 'kpc')
    bbox_lim = box_len.in_units('code_length')
    bounding_box = [[center[0]-bbox_lim,center[0]+bbox_lim],
            [center[1]-bbox_lim,center[1]+bbox_lim],
            [center[2]-bbox_lim,center[2]+bbox_lim]]

    left = np.array([pos[0] for pos in bounding_box])
    right = np.array([pos[1] for pos in bounding_box])
    octree = ds.octree(left,right,n_ref=32)
    #print('------ constructing octree ------ ')
    ds.parameters['octree'] = octree

    try:
        ds.add_field(('gas','smoothedmasses'), function=_gassmoothedmasses, sampling_type='particle',units='g', particle_type=True)
    except:
        return -1,-1
    gmass = ds.all_data()[('gas', 'smoothedmasses')]
    #gmass_sm.append(np.sum(gmass.value))
    #gmass_p.append(np.sum(ds.all_data()["PartType0", "Masses"].in_units("g").value))

    return np.sum(gmass.value), np.sum(ds.all_data()["PartType0", "Masses"].in_units("g").value)



with Pool(20) as p:
    out1, out2 = zip(*tqdm(p.imap(open_snaps, np.arange(300)), total=300))
    gmass_sm = np.ravel(out1)
    gmass_p = np.ravel(out2)

np.savez('/orange/narayanan/s.lower/simba/snap305_gas_masses.npz', smoothed=gmass_sm, particle=gmass_p)
