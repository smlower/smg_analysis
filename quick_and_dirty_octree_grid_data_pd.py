#from powderday.analytics import dump_data
import ipdb
from hyperion.model import ModelOutput
from hyperion.grid.yt3_wrappers import find_order
import astropy.units as u
#import powderday.config as cfg
import sys, yt
from tqdm.auto import tqdm
import numpy as np

yt.set_log_level(100)

print('\n\n----------------------\n Getting grid dust properties \n----------------------')

#run=np.arange(32)#int(sys.argv[1])

for run in [18]:
    if run in [0,2,3,4,5,6,8,13,19]:
        path = f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run{run}_halo0/'
        end = 84
        galaxy = 0
        z6p4 = 42
        start = 5
        deltat = 1.5e7
        snaps =np.arange(49,18,-1)
        chosen_snaps = [19,21,24,27,30,34,39,44,49]
    else:
        path= f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/z10_nodust_track_output/run{run}_halo0/'
        end = 2
        galaxy = 0
        start = 0
        z6p4 = 19
        dust_scale = 3.0
        snaps =np.arange(23,-1,-1)
        chosen_snaps = [0,2,4,6,8,11,15,19,23]
        
    galaxy = run
    snap=chosen_snaps[-2]
    pd_dir = f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/pd_runs/run{run}_halo0/snap{snap}/'
    if galaxy == 21:
        continue
    sys.path.insert(0, pd_dir)
    par = __import__('highz_nebular_parameters_master')
    model = __import__(f'run{galaxy}_snap{snap}_galaxy0')
    cfg_par = par
    cfg_model = model
    fname = path+f'/snapshot_{snap:03d}.hdf5'
    ds = yt.load(fname)
    data = ds.all_data()
    center = [cfg_model.x_cent,cfg_model.y_cent,cfg_model.z_cent]
    box_len = cfg_par.zoom_box_len
    box_len = ds.quan(box_len,'kpc')
    box_len = float(box_len.to('code_length').value)
    bbox_lim=box_len
    bounding_box = [[center[0]-bbox_lim,center[0]+bbox_lim],
            [center[1]-bbox_lim,center[1]+bbox_lim],
            [center[2]-bbox_lim,center[2]+bbox_lim]]

    left = np.array([pos[0] for pos in bounding_box])
    right = np.array([pos[1] for pos in bounding_box])
    octree = ds.octree(left,right,n_ref=cfg_par.n_ref) 
    ds.parameters['octree'] = octree

    def _dustsmoothedmasses(field, data):
        dsm = ds.arr(data.ds.parameters['octree'][('PartType0','Dust_Masses')],'code_mass')
        return dsm

    ds.add_field(('dust','smoothedmasses'), function=_dustsmoothedmasses, sampling_type='particle',units='code_mass')#, particle_type=True)


    dmass = ds.all_data()[('dust', 'smoothedmasses')]

    m = ModelOutput(pd_dir+f'/run{run}_snap{snap:03d}_galaxy0.rtout.sed')
    grid = m.get_quantities()
    order = find_order(grid.refined)
    refined = grid.refined[order]
    quantities = {}
    for field in grid.quantities:
        quantities[('gas', field)] = np.atleast_2d(grid.quantities[field][0][order][~refined]).transpose()

    #try:
    dust_temp = quantities['gas','temperature']*u.K
    #except:
    #    continue
    outfile = pd_dir+f'/grid_physical_properties.{snap:03d}_run{run}_galaxy0_dust.npz'
    print('dumping data')
    np.savez(outfile, grid_dust_mass=dmass,grid_dust_temp=dust_temp)   

