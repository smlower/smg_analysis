import sphviewer as sph
import numpy as np
import yt, caesar, h5py
import pandas as pd
import sys
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
matplotlib.rcParams.update({
    "savefig.facecolor": "w",
    "figure.facecolor" : 'w',
    "figure.figsize" : (10,8),
    "text.color": "k",
    "legend.fontsize" : 20,
    "font.size" : 30,
    "axes.edgecolor": "k",
    "axes.labelcolor": "k",
    "axes.linewidth": 3,
    "xtick.color": "k",
    "ytick.color": "k",
    "xtick.labelsize" : 25,
    "ytick.labelsize" : 25,
    "ytick.major.size" : 12,
    "xtick.major.size" : 12,
    "ytick.major.width" : 2,
    "xtick.major.width" : 2,
    "font.family": 'STIXGeneral',
    "mathtext.fontset" : "cm"
})


### convience functions ###

def idmatch(id_i, id_f):
    ## match items in set (i) to those in set (f)
    index_dict_i = dict((k,i) for i,k in enumerate(id_i));
    index_dict_f = dict((k,i) for i,k in enumerate(id_f));
    inter = set(id_i).intersection(set(id_f));
    indices_i = np.array([ index_dict_i[x] for x in inter ]);
    indices_f = np.array([ index_dict_f[x] for x in inter ]);
    return indices_i, indices_f;

def compile_matched_ids( id_i, id_f ):
    sub_id_i, sub_id_f = idmatch(id_i,id_f)
    nomatch_from_i_in_f = (id_i > -1.0e40) 
    if (sub_id_i.size > 0):
        nomatch_from_i_in_f[sub_id_i] = False
    nomatch_from_f_in_i = (id_f > -1.0e40)
    if (sub_id_f.size > 0):
        nomatch_from_f_in_i[sub_id_f] = False
    return sub_id_i, sub_id_f


# PUT IT ALL TOGETHER !!!!!
nframes = 3
frame_count = 0
extent = 10
this_snap = int(sys.argv[1])

if this_snap == 27:
    frame_count = 0
else:
    frame_count += nframes*(this_snap-27)
for snap in [this_snap]:
    ds_i = h5py.File(f'/blue/narayanan/s.lower/zoom_temp/run2_halo0_ml09_3myr/snapshot_{snap:03d}.hdf5','r')
    obj_i = caesar.load(f'/blue/narayanan/s.lower/zoom_temp/run2_halo0_ml09_3myr/Groups/caesar_snapshot_{snap:03d}.hdf5')
    obj_f = caesar.load(f'/blue/narayanan/s.lower/zoom_temp/run2_halo0_ml09_3myr/Groups/caesar_snapshot_{snap+1:03d}.hdf5')
    ds_f = h5py.File(f'/blue/narayanan/s.lower/zoom_temp/run2_halo0_ml09_3myr/snapshot_{snap+1:03d}.hdf5','r')
    print('matching IDs')
    ids_i, ids_f = np.array(ds_i['PartType0']['ParticleIDs']), np.array(ds_f['PartType0']['ParticleIDs'])
    matched_ids_i, matched_ids_f = compile_matched_ids(ids_i,ids_f)
    print('loading positions & masses')
    halo_pos_i = np.array(ds_i['PartType0']['Coordinates'])[matched_ids_i]
    halo_pos_f = np.array(ds_f['PartType0']['Coordinates'])[matched_ids_f]

    halo_mass_i = np.array(ds_i['PartType0']['Masses'])[matched_ids_i]
    halo_mass_f = np.array(ds_f['PartType0']['Masses'])[matched_ids_f]

    x_i,y_i,z_i = halo_pos_i[:,0],halo_pos_i[:,1],halo_pos_i[:,2]
    x_f, y_f, z_f = halo_pos_f[:,0], halo_pos_f[:,1], halo_pos_f[:,2]
    
    com_i = obj_i.halos[0].minpotpos.in_units('code_length').value
    com_f = obj_f.halos[0].minpotpos.in_units('code_length').value
    
    print('building mass and position interpolation lists')
    pos_masterlist = np.empty((nframes+2,len(matched_ids_f), 3))
    mass_masterlist = np.empty((nframes+2,len(matched_ids_f)))
    com_masterlist = np.empty((nframes+2,3))
    mass_interpd = np.linspace(halo_mass_i, halo_mass_f, nframes+2)
    com_interpd = np.linspace(com_i, com_f, nframes+2)
    for component in [0,1,2]:
        init_pos = halo_pos_i[:,component]
        final_pos = halo_pos_f[:,component]
        #print(init_pos)
        pos_interpd = np.linspace(init_pos, final_pos, nframes+2)
        #print(pos_interpd)
        for frame in range(nframes):
            pos_masterlist[frame,:,component] = pos_interpd[frame]
            mass_masterlist[frame,:] = mass_interpd[frame]
            com_masterlist[frame,:] = com_interpd[frame]
    print('generating images')
    for frame in range(nframes):
        halos_pos = pos_masterlist[frame]
        halos_pmass = mass_masterlist[frame]
        hcoord = com_masterlist[frame]
        P = sph.Particles(halos_pos*0.68, halos_pmass*0.68)
        C = sph.Camera(x=hcoord[0]*0.68, y=hcoord[1]*0.68, z=hcoord[2]*0.68,
                       r=60, zoom=2,
                       t=30, p=30, roll=30,
                       extent=[-extent,extent,-extent,extent],
                       xsize=400, ysize=400)
        S = sph.Scene(P, Camera=C)
        R = sph.Render(S)                                                                                                                               
        vmax = 1e1
        vmin = 1e-4
        img = R.get_image()
        img[img == 0] = vmin
        cNorm  = colors.LogNorm(vmin=vmin,vmax=vmax)
        sm1 = plt.imshow(img, extent=[-extent,extent,-extent,extent],
                  cmap=cm.bone, norm=cNorm)
        plt.savefig(f'/blue/narayanan/s.lower/zoom_vis/interp_3myr_frame{frame_count}.png', dpi=100)
        print(f'done with snap {snap} frame {frame+1}')
        frame_count += 1
    ds_i.close()
    ds_f.close()
