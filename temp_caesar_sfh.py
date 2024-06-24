import sys, os
from tqdm.auto import tqdm
import fsps
import yt
import caesar
import numpy as np
import pickle


# Right now, getting SFH from snapshot 25, z=5.8

print('Loading fsps')
fsps_ssp = fsps.StellarPopulation(sfh=0,
                zcontinuous=1,
                imf_type=2,
                zred=0.0, add_dust_emission=False)
solar_Z = 0.0142

for i in [2]:
    print('--------------------')
    print(f'Loading Run {i}')
    if i == 0:
        path = f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output//run{i}_halo0/'
        print(path)
        obj = caesar.load(path+'/Groups/caesar_snapshot_029.hdf5')
        ds = yt.load(path+'snapshot_029.hdf5')
    else:
        try:
            path = f'/blue/narayanan/s.lower/zoom_temp/run{i}_halo0/'
            print(path)
            obj = caesar.load(path+'/Groups/caesar_snapshot_025.hdf5')
            ds = yt.load(path+'/snapshot_025.hdf5')
        except:
            path = f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run{i}_halo0_10myr/'
            print(path)
            obj = caesar.load(path+'/Groups/caesar_snapshot_069.hdf5')
            ds = yt.load(path+'/snapshot_069.hdf5')
    obj.yt_dataset = ds
    dd = obj.yt_dataset.all_data()
    print('    Loading particle data')
    scalefactor = dd[("PartType4", "StellarFormationTime")]
    stellar_masses = dd[("PartType4", "Masses")]
    stellar_metals = dd[("PartType4", 'metallicity')]

    # Compute the age of all the star particles from the provided scale factor at creation                               
    formation_z = (1.0 / scalefactor) - 1.0
    yt_cosmo = yt.utilities.cosmology.Cosmology(hubble_constant=0.68, omega_lambda = 0.7, omega_matter = 0.3)
    stellar_formation_times = yt_cosmo.t_from_z(formation_z).in_units("Gyr")

    # Age of the universe right now                                                                                     
    simtime = yt_cosmo.t_from_z(ds.current_redshift).in_units("Gyr")
    stellar_ages = (simtime - stellar_formation_times).in_units("Gyr")

    #want galaxy 0 SFH
    slist = obj.galaxies[0].slist
    this_galaxy_stellar_ages = stellar_ages[slist] # this takes all the stellar ages from the entire sim and gets just those for this galaxy, i think
    this_galaxy_stellar_masses = stellar_masses[slist]
    this_galaxy_stellar_metals = stellar_metals[slist]
    this_galaxy_formation_masses = []
    print('    finding initial particle mass')
    for age, metallicity, mass in zip(this_galaxy_stellar_ages, this_galaxy_stellar_metals, this_galaxy_stellar_masses):
        mass = mass.in_units('Msun')
        fsps_ssp.params['logzsol'] = np.log10(metallicity/solar_Z)
        fsps_ssp.params['tage'] = age
        mass_remaining = fsps_ssp.stellar_mass
        massform = mass / mass_remaining
        this_galaxy_formation_masses.append(massform)
    
    this_galaxy_formation_masses = np.array(this_galaxy_formation_masses)
    this_galaxy_formation_times = np.array(simtime - this_galaxy_stellar_ages, dtype=float)
    
    outfile = f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/galaxy_properties/sfh_m25zoom_run{i}_10myr.pickle' 
    with open(outfile, 'wb') as f:
        pickle.dump({'massform':this_galaxy_formation_masses,'tform':this_galaxy_formation_times},f)
        
        
