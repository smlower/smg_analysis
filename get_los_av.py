import numpy as np
import yt, caesar, h5py
from caesar.pyloser import pyloser


ds = yt.load('/orange/narayanan/s.lower/simba/m25n256_dm/zooms/z10_nodust_track_output/run2_halo0_10myr/snapshot_065.hdf5')
#ad = ds.all_data()
obj = caesar.load('/orange/narayanan/s.lower/simba/m25n256_dm/zooms/z10_nodust_track_output/run2_halo0_10myr/Groups/caesar_snapshot_065.hdf5')
obj.yt_dataset = ds
dd = obj.yt_dataset.all_data()
#obj._kwargs=None
print('initializing pyloser')

for direction in ['x', 'y', 'z']:

    pylose = pyloser.photometry(obj, [obj.galaxies[0]], ds=ds, ssp_table_file='SSP_Kroupa_EL.hdf5', view_dir=direction,nproc=25)


    print('running pyloser in '+str(direction)+' direction')
    test = pylose.run_pyloser()

    print('getting group Av list')
    #av_per_group = []
    av_per_group = pylose.groups[0].group_Av

    print('done. saving')
    np.savez('/orange/narayanan/s.lower/simba/m25n256_dm/zooms/galaxy_properties/run2_snap065_LOS_'+direction+'dir.npz', Av=av_per_group)
