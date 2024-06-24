from powderday.analytics import dump_data
from hyperion.model import ModelOutput
from hyperion.grid.yt3_wrappers import find_order
import astropy.units as u
import powderday.config as cfg
import sys, yt
import numpy as np

#yt.set_log_level(50)

print('\n\n----------------------\n Getting grid dust properties \n----------------------')

galaxy=int(sys.argv[1])
snap=74
pd_dir = '/orange/narayanan/d.zimmerman/simba/m25n512/snap74/pd_scripts/'

sys.path.insert(0, pd_dir)
par = __import__('parameters_master_catalog')
model = __import__(f'snap{snap}_galaxy{galaxy}')
cfg.par = par
cfg.model = model

fname = f'/orange/narayanan/d.zimmerman/simba/m25n512/snap74/filtered/galaxy_{galaxy}.hdf5'
ds = yt.load(fname)
data = ds.all_data()
center = [cfg.model.x_cent,cfg.model.y_cent,cfg.model.z_cent]
box_len = cfg.par.zoom_box_len
box_len = ds.quan(box_len,'kpc')
box_len = float(box_len.to('code_length').value)
bbox_lim=box_len
bounding_box = [[center[0]-bbox_lim,center[0]+bbox_lim],
            [center[1]-bbox_lim,center[1]+bbox_lim],
            [center[2]-bbox_lim,center[2]+bbox_lim]]

particle_dmass = data.ds.arr(data[("PartType0", "Dust_Masses")].value, 'code_mass')
print(np.sum(particle_dmass.in_units('Msun')))
left = np.array([pos[0] for pos in bounding_box])
right = np.array([pos[1] for pos in bounding_box])
octree = ds.octree(left,right,n_ref=cfg.par.n_ref)
ds.parameters['octree'] = octree

def _dustsmoothedmasses(field, data):
    dsm = ds.arr(data.ds.parameters['octree'][('PartType0','Dust_Masses')].value,'code_mass')
    return dsm

ds.add_field(('dust','smoothedmasses'), function=_dustsmoothedmasses, sampling_type='particle',units='code_mass', particle_type=True)


dmass = ds.all_data()[('dust', 'smoothedmasses')].in_units('Msun')
print(np.sum(dmass))
pd_dir = '/orange/narayanan/d.zimmerman/simba/m25n512/snap74/pd_runs/'
m = ModelOutput(pd_dir+f'/snap{snap}.galaxy{galaxy}.rtout.sed')
#m = ModelOutput(pd_dir+f'/run{run}_snap{snap:03d}.rtout.image')       
grid = m.get_quantities()
order = find_order(grid.refined)
refined = grid.refined[order]

quantities = {}
for field in grid.quantities:
    quantities[('gas', field)] = np.atleast_2d(grid.quantities[field][0][order][~refined]).transpose()

print(quantities.keys())
dens = quantities["gas","smootheddensity"]
dust_temp = quantities['gas','temperature']*u.K

outfile = f'/orange/narayanan/s.lower/simba/galaxy_properties/grid_dust_properties/m25n512_snap74/grid_physical_properties_galaxy{galaxy}_dust.npz'
print('dumping data')
np.savez(outfile, grid_dust_mass=dmass,grid_dust_temp=dust_temp, density=dens)   

