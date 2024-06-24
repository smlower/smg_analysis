import numpy as np
import caesar, yt
from caesar.periodic_kdtree import PeriodicCKDTree
import sys
from yt.funcs import mylog
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

halo_num = int(sys.argv[1])
print('reading snap')
ds = yt.load('/orange/narayanan/s.lower/simba/m25n256_dm/output/run0/snapshot_006.hdf5')
ad = ds.all_data()
print('reading caesar file')
obj = caesar.load('/orange/narayanan/s.lower/simba/m25n256_dm/output/run0/Groups/caesar_snapshot_006.hdf5')
print('reading ICs ds')
ic_ds = yt.load('/orange/narayanan/s.lower/simba/m25n256_dm/IC_stuff/ics_m25n256_Run0.0')
ic_ad = ic_ds.all_data()

z = obj.simulation.redshift
outfile = f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/halo{halo_num}_mask.txt'
#print('writing mask')
#obj.halos[0].write_IC_mask(ic_ds,outfile, radius_type='dm_r80')


print(f'finding halo {halo_num} particles within 2.5*r_80 radius at z={z}')
print(f'halo {halo_num} has {len(obj.halos[halo_num].dmlist)} DM particles')
#halo_dmpids = ad['PartType1', 'ParticleIDs']#[obj.halos[halo_num].dmlist]
halo_dmpos = ad['PartType1', 'Coordinates']#[obj.halos[halo_num].dmlist]
box    = ic_ds.domain_width[0].d
bounds = np.array([box,box,box])
print('    constructing tree')
dm_TREE = PeriodicCKDTree(bounds, halo_dmpos)
dm_within_radius = dm_TREE.query_ball_point(obj.halos[halo_num].pos.in_units('code_length'), obj.halos[halo_num].radii['total_r80'].in_units('code_length') * 2.5)

print('matching particles in ICs')
#halo_mask = ad['PartType1', 'ParticleIDs'][dm_within_radius].d
#ic_dmpids = ic_ad['PartType1', 'ParticleIDs'].d
#ic_dmpos = ic_ad['PartType1', 'Coordinates'].in_units('code_length')

halo_mask = ad['PartType1', 'ParticleIDs'][dm_within_radius].d       
ic_dmpids = ic_ad['Halo', 'ParticleIDs'].d                                                                                                             
ic_dmpos = ic_ad['Halo', 'Coordinates'].in_units('code_length') 
print('    finding matches')
matches  = np.in1d(ic_dmpids, halo_mask, assume_unique=True)

nmatches = len(np.where(matches)[0])
nvalid   = len(dm_within_radius)

if nmatches != nvalid:
    print(f'        could not match all particles in z={z} to snapshot 0')
else: print('        matched all particles')
matched_pos = ic_dmpos[matches].in_units('code_length') / box
print(f'        number of matched particles: {len(matched_pos)}')
print(f'writing mask for halo {halo_num} particles')
f = open(outfile, 'w')
for i in range(0, len(matched_pos)):
    f.write('%e %e %e\n' % (matched_pos[i,0], matched_pos[i,1], matched_pos[i,2]))
f.close()


print(f'Done. Mask at {outfile}')
