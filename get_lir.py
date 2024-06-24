from multiprocessing import Pool
from hyperion.model import ModelOutput
import astropy.units as u
import astropy.constants as constants
from astropy.cosmology import FlatLambdaCDM
import sphviewer as sph
import numpy as np
import yt, caesar, h5py
import pandas as pd
from tqdm.auto import tqdm
import scipy.stats
import sys

cosmo = FlatLambdaCDM(H0=68, Om0=0.3, Tcmb0=2.725)
yt_cosmo = yt.utilities.cosmology.Cosmology(hubble_constant=0.68, omega_lambda = 0.7, omega_matter = 0.3)


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def get_pd_sed(file, z):
    m = ModelOutput(file)
    wav, flx = m.get_sed(inclination='all', aperture=-1)                                                                                                                                  
    wave  = np.asarray(wav)[::-1]*u.micron  
    lir = []
    s850 = []
    w850 = find_nearest(wave.value, 1400/(1.+z))
    ir1 = 8*u.micron
    ir2 = 1000*u.micron
    for i in range(5):
        flux = np.asarray(flx[i])[::-1]*u.erg/u.s
        s850.append(flux[w850].value)
        lir.append(np.trapz(flux[find_nearest(wave, ir1):find_nearest(wave, ir2)]/wave[find_nearest(wave, ir1):find_nearest(wave, ir2)],wave[find_nearest(wave, ir1):find_nearest(wave, ir2)])/ constants.L_sun.cgs)
    #lir = np.trapz(flux[find_nearest(wave, ir1):find_nearest(wave, ir2)]/,wave[find_nearest(wave, ir1):find_nearest(wave, ir2)])/ constants.L_sun.cgs
    
    return wav, flux, np.max(np.log10(lir)), np.max(s850)/constants.L_sun.cgs.value



def get_lir_times(i):
    for run in [i]:
        if run == 19:
            path = f'/blue/narayanan/s.lower/zoom_temp/run19_halo0/'
            end = 43
            galaxy = 0
            start = 6
            deltat = 15
        elif run == 8:
            path = f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run8_halo0/'
            end = 78
            galaxy = 0
            start = 22
            deltat = 15
        elif run == 13:
            path = f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run13_halo0/'
            end = 88
            galaxy = 0
            start = 22
            deltat = 15
        elif run == 5:
            path = f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run5_halo0/'
            end = 86
            #start = 
            start = 16
            galaxy = 0
            deltat = 15
        elif run == 0:
            path= f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/z10_nodust_track_output/run{run}_halo0/'
            end = 34
            galaxy = 0
            start = 0
            deltat = 15
        elif run == 2:
            path= f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/z10_nodust_track_output//run{run}_halo0_10myr/'
            end = 69
            galaxy = 0
            start = 10
            deltat = 10
        elif run == 25:
            path= f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/z10_nodust_track_output/run{run}_halo0/'
            end = 25
            galaxy = 0
            start = 10
            deltat = 30
        else:
            path= f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/z10_nodust_track_output/run{run}_halo0/'
            end = 25
            galaxy = 0
            start = 0
            deltat = 30
        this_gal_lir = []
        this_gal_s850 = []
        time = []
        this_gal_smg_time, this_gal_ulirg_time = 0,0
        for i in tqdm(range(start, end)):
            #print(end)
            #print(path)
            path = f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/pd_runs/run{run}_halo0/snap{i}/' 
            sys.path.append(path)
            try:
                run_info = __import__(f'run{run}_snap{i}')
            except:
                continue
            redshift = (run_info.TCMB / 2.73) - 1
            #obj = caesar.load(path+f'/Groups/caesar_snapshot_{i:03d}.hdf5')
            pd_path = f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/pd_runs/run{run}_halo0/snap{i}/run{run}_snap{i:03d}.rtout.sed'
            try:
#                obj = caesar.load(path+f'/Groups/caesar_snapshot_{i:03d}.hdf5')
                _, _, lir, s_850 = get_pd_sed(pd_path, redshift)
                this_gal_lir.append(lir)
                this_gal_s850.append(s_850)
            except:
                continue
            props = np.load(f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/pd_runs/run{run}_halo0/snap{i}/grid_physical_properties.{i:\
03d}_galaxy0.npz', allow_pickle=True)
            dustmass = np.sum(props['particle_dustmass'])
            dl = yt_cosmo.luminosity_distance(0,redshift).in_units('cm').value
            lum_lim = 5 * 1e-23 * (4 * np.pi * dl**2) * (constants.c.cgs.value / 0.085) / 1000
            smg_cutoff = np.log10(lum_lim / constants.L_sun.cgs.value)
            time.append(yt_cosmo.t_from_z(redshift).in_units('Myr').value)
            if np.log10(s_850) > smg_cutoff:
                this_gal_smg_time += deltat

            if lir > 12:
                this_gal_ulirg_time += deltat

    return this_gal_smg_time, this_gal_ulirg_time, this_gal_lir, time, this_gal_s850, dustmass



if __name__ == '__main__':

    run = int(sys.argv[1])

    smg_time, ulirg_time, lir, time, s850, final_dustmass = get_lir_times(run)
    print(f'smg time: {smg_time}')
    print(f'ulirg time: {ulirg_time}')
    print(f'lir: {lir}')

    np.savez(f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/galaxy_properties/run{run}_lir.npz', lir=lir, smg_time=smg_time, ulirg_time=ulirg_time, time=time, s850=s850, final_dustmass=final_dustmass)
