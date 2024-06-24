import sys, os
from tqdm.auto import tqdm
import fsps
import yt
import caesar
from caesar.periodic_kdtree import PeriodicCKDTree
import numpy as np
import pickle
from multiprocessing import Pool
import itertools

#run = int(sys.argv[1])
print('Loading fsps')
fsps_ssp = fsps.StellarPopulation(sfh=0,
                zcontinuous=1,
                imf_type=2,
                zred=0.0, add_dust_emission=False)
solar_Z = 0.0142

final_massfrac, final_formation_times, final_formation_masses = [], [], []

def load_data(run,galaxy=None):
    print('loading data')
    print(galaxy)
    if run == 19:
        path = f'/blue/narayanan/s.lower/zoom_temp/run19_halo0/'
        end = 47
#        galaxy = 0
    elif run == 8:
        #path= f'/blue/narayanan/s.lower/zoom_temp/run8_restart/'
        path= f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run{run}_halo0/'
        end = 40 #89
        z6p4 = 42
        galaxy = 0
    elif run == 13:
        path= f'/blue/narayanan/s.lower/zoom_temp/run13_halo0_restart/'
        #path= f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run{run}_halo0/'
        end = 90
        galaxy = 0
    elif run == 5:
        path= f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run{run}_halo0/'
        end = 86
        galaxy = 0

    elif run == 0:
        path= f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run{run}_halo0/'
        end = 29
        galaxy = 0
    elif run == 2:
        path= f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run{run}_halo0_10myr/'
        end = 69
        galaxy = 0
    elif run == 'FIRE':
        path = '/blue/narayanan/s.lower/zoom_temp/run8_halo0_FIRE/'
        end = 53
        galaxy = 0
    elif run == 'z2_48':
        path = f'/orange/narayanan/s.lower/simba/desika_filtered_snaps/snap160/galaxy_48.hdf5'
    elif run == 'run8_smoothing_length':
        path = '/blue/narayanan/s.lower/zoom_temp/run8_smoothing_length/'
        end = 20
    else:
        path= f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/z10_nodust_track_output/run{run}_halo0/'
        end = 25
        galaxy = 0
    snap = end
    galaxy = galaxy
    obj = caesar.load(path+f'/Groups/caesar_snapshot_{snap:03d}.hdf5')
    ds = yt.load(path+f'snapshot_{snap:03d}.hdf5')
    #obj = caesar.load('/orange/narayanan/desika.narayanan/gizmo_runs/simba/m25n512/output/Groups/caesar_0160_z2.000.hdf5')
#    ds = yt.load(path)
    obj.yt_dataset = ds
    dd = obj.yt_dataset.all_data()

    halo_starpos = dd['PartType4', 'Coordinates'].in_units('kpc')
    box    = ds.domain_width[0].in_units('kpc')
    bounds = np.array([box,box,box])
    print('finding stars within radius')
    center = obj.galaxies[galaxy].pos.in_units('kpc')
    print(f"t_H = {obj.simulation.time.in_units('Myr')}")
    star_TREE = PeriodicCKDTree(bounds, halo_starpos)
    star_within_radius = star_TREE.query_ball_point(center, ds.quan(60., 'kpc'))
    #star_within_radius = obj.halos[0].slist
    scalefactor = dd[("PartType4", "StellarFormationTime")][star_within_radius]
    stellar_masses = dd[("PartType4", "Masses")][star_within_radius].in_units('Msun')
    stellar_metals = dd[("PartType4", 'metallicity')][star_within_radius]

    # Compute the age of all the star particles from the provided scale factor at creation                                               
    formation_z = (1.0 / scalefactor) - 1.0
    yt_cosmo = yt.utilities.cosmology.Cosmology(hubble_constant=0.68, omega_lambda = 0.7, omega_matter = 0.3)
    stellar_formation_times = yt_cosmo.t_from_z(formation_z).in_units("Gyr")

    simtime = yt_cosmo.t_from_z(ds.current_redshift).in_units("Gyr")
    stellar_ages = (simtime - stellar_formation_times).in_units("Gyr")

    return stellar_ages, stellar_masses, stellar_metals, simtime


def get_sfh(this_galaxy_stellar_ages, this_galaxy_stellar_masses,this_galaxy_stellar_metals, simtime):
    metallicity = this_galaxy_stellar_metals#[star]
    age = this_galaxy_stellar_ages#[star]
    mass = this_galaxy_stellar_masses#[star]
    mass = mass.in_units('Msun')
    fsps_ssp.params['logzsol'] = np.log10(metallicity/solar_Z)
    mass_remaining = fsps_ssp.stellar_mass
    initial_mass = np.interp(np.log10(age*1e9), fsps_ssp.ssp_ages, mass_remaining)
    massform = mass / initial_mass
    this_galaxy_formation_masses = massform
    this_galaxy_formation_masses = np.array(this_galaxy_formation_masses)
    this_galaxy_formation_times = np.array(simtime - age, dtype=float)
    return this_galaxy_formation_times, this_galaxy_formation_masses

def func_star(params):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return get_sfh(*params)


with Pool(32) as p:
    for run, galaxy in zip([19,19], [0,1]):
        stellar_ages, stellar_masses, stellar_metals, simtime = load_data(run=run,galaxy=galaxy)
        print('entering pool for SFH calculation')
        out1, out2 = zip(*tqdm(p.imap(func_star, zip(stellar_ages, stellar_masses, stellar_metals, itertools.repeat(simtime))), total=len(stellar_ages)))
        final_formation_times = np.ravel(out1)
        final_formation_masses = np.ravel(out2)
        if galaxy is not None:
            outfile = f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/galaxy_properties/sfh_m25zoom_run{run}_within_50kpc_galaxy{galaxy}_{simtime}.pickle'
        else:
            outfile = f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/galaxy_properties/sfh_m25zoom_run{run}_within_50kpc_{simtime}.pickle'
        with open(outfile, 'wb') as f:
            pickle.dump({'massform':final_formation_masses,
                         'tform':final_formation_times},f)

