import sphviewer as sph
import numpy as np
import yt, caesar, h5py
import sys
import pandas as pd
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


frame = int(sys.argv[1])

vmax = 1e1
vmin = 1e-4

extent = 10

p = np.linspace(60,90,70)
t = np.linspace(30,90,70)
roll = np.linspace(30,180,70)
r = np.linspace(30,30,70)



import scipy

fig = plt.figure(figsize=(16,12))
ax1 = plt.subplot2grid((3,3), (1, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid((3,3), (1, 2))
ax3 = plt.subplot2grid((3,3), (2,2))

extent = 10

def get_massform(massform):
    return np.sum(massform) / (binwidth * 1e9)

p = np.linspace(30,90,70)
t = np.linspace(30,90,70)
roll = np.linspace(30,180,70)
r = np.linspace(30,30,70)

dat = pd.read_pickle(f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/galaxy_properties/sfh_m25zoom_run2_10myr.pickle')
massform = np.array(dat['massform'])
tform = np.array(dat['tform'])


mdust = []
time = []

if frame == 0:
    obj = caesar.load(f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run2_halo0_10myr/Groups/caesar_snapshot_020.hdf5')
    mdust.append(obj.galaxies[0].masses['dust'].in_units('Msun').value)
    time.append(obj.simulation.time.in_units('Gyr').value)
else:
    for snapnum in range(0,frame):
        snap = np.arange(20,70,1)
        obj = caesar.load(f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run2_halo0_10myr/Groups/caesar_snapshot_{snap[snapnum]:03d}.hdf5')
        mdust.append(obj.galaxies[0].masses['dust'].in_units('Msun').value)
        time.append(obj.simulation.time.in_units('Gyr').value)

for i in [frame]:
    snapnum = np.arange(20,70,1)    
    ds = h5py.File(f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run2_halo0_10myr/snapshot_{snapnum[i]:03d}.hdf5','r')
    obj = caesar.load(f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run2_halo0_10myr/Groups/caesar_snapshot_{snapnum[i]:03d}.hdf5')
    halos_pos = np.array(ds['PartType0']['Coordinates'])[obj.halos[0].glist]
    halos_pmass = np.array(ds['PartType0']['Masses'])[obj.halos[0].glist]
    hcoord = obj.halos[0].minpotpos.in_units('code_length').value
    P = sph.Particles(halos_pos*0.68, halos_pmass*0.68)
    C = sph.Camera(x=hcoord[0]*0.68, y=hcoord[1]*0.68, z=hcoord[2]*0.68,
                   r=60, zoom=2,
                   t=30, p=p[i], roll=30,
                   extent=[-extent,extent,-extent,extent],
                   xsize=500, ysize=500)
    S = sph.Scene(P, Camera=C)
    R = sph.Render(S)                                                                                                                                                             
    img = R.get_image()
    img[img == 0] = vmin
    cNorm  = colors.LogNorm(vmin=vmin,vmax=vmax)
    ax1.imshow(img, extent=[-extent,extent,-extent,extent],
              cmap=cm.bone, norm=cNorm)
  
    ax1.axis('off')
    
    t_H = obj.simulation.time.in_units('Gyr').value #Gyr
    #print(t_H)
    binwidth = 0.01
    bins = np.arange(0, t_H, binwidth) 
    sfrs, bins, binnumber = scipy.stats.binned_statistic(tform, massform, statistic=get_massform, bins=bins)
    sfrs[np.isnan(sfrs)] = 0
    bincenters = 0.5*(bins[:-1]+bins[1:])
    ax2.plot(bincenters, sfrs)
    ax2.set_xlim([0.6,1.2])
    ax2.set_ylabel('SFR [$\mathrm{M}_{\odot} \mathrm{yr}^{-1}$]')
    ax2.set_xticks([])
    ax3.plot(time, mdust)
    ax3.set_yscale('log')
    ax3.set_ylabel('M$_\mathrm{d}$ [$\mathrm{M}_{\odot}$]')
    ax3.set_xlabel('t$_H$ [Gyr]')
    ax3.set_xlim([0.6,1.2])

    
    
    plt.savefig(f'/home/s.lower/scripts/smg_movie/run2_sphview/frame{i}_snap{i}_sfr_mdust.png', dpi=200,bbox_inches='tight')
