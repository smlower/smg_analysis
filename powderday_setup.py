#purpose: to set up slurm files and model *.py files from the
#positions written by caesar_cosmology_npzgen.py for a cosmological
#simulation.  This is written for the University of Florida's
#HiPerGator2 cluster.

import numpy as np
from subprocess import call
import sys
import caesar
nnodes=1

model_dir_base = '/orange/narayanan/s.lower/simba/m25n256_dm/zooms/pd_runs/birth_clouds/'

#################
#starburst_galaxies = [0,1,2,3,6,7,8,16,18,19,22,23]
for run in range(19,32):
#        if run == 19:
#                path = f'/blue/narayanan/s.lower/zoom_temp/run19_halo0/'
#                end = 44
#                start = 5
#                deltat = 1.5e7
#                z6p4 = 42
#                dust_scale = 3.0
#                snaps = np.arange(49,18,-1)
#                chosen_snaps = [19,21,24,27,30,34,39,44,49]
        if run in [0,2,3,4,5,6,8,13,19]:
                path = f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run{run}_halo0/'
                end = 84
                galaxy = 0
                z6p4 = 42
                start = 5
                deltat = 1.5e7
                snaps =np.arange(49,18,-1)
                chosen_snaps = [19,21,24,27,30,34,39,44,49]
        elif run == 21:
                continue
        else:
                path= f'/orange/narayanan/s.lower/simba/m25n256_dm/zooms/z10_nodust_track_output/run{run}_halo0/'
                end = 2
                galaxy = 0
                start = 0
                z6p4 = 19
                dust_scale = 3.0
                snaps =np.arange(23,-1,-1)
                chosen_snaps = [0,2,4,6,8,11,15,19,23]
        if run == 19:
            for galaxy in [0,1]:
                    for snap_num in snaps:
                            thissnap = path+f"/snapshot_{snap_num:03d}.hdf5"
                            thisobj = path+f"/Groups/caesar_snapshot_{snap_num:03d}.hdf5"
                            model_run_name=f'run{run}_snap{snap_num}_galaxy{galaxy}'
                            obj = caesar.load(thisobj)
                            pos = np.array(obj.galaxies[galaxy].pos.in_units('code_length'))
                            #progens = np.min(obj.galaxies[galaxy].progen_galaxy_star)
                            #galaxy = progens
                            xpos, ypos, zpos = pos[0],pos[1],pos[2]
                            snap_redshift = obj.simulation.redshift
                            tcmb = 2.73*(1.+snap_redshift)
                            if snap_num in chosen_snaps:
                                    cmd = "./generate_model_files.sh "+str(model_dir_base)+' '+str(run)+' '+str(path)+' '+str(snap_num)+' '+str(model_run_name)+' '+str(galaxy)+' '+str(xpos)+' '+str(ypos)+' '+str(zpos)+' '+str(tcmb)
                                    call(cmd,shell=True)

        else:
                galaxy = 0
                for snap_num in snaps:
                            thissnap = path+f"/snapshot_{snap_num:03d}.hdf5"
                            thisobj = path+f"/Groups/caesar_snapshot_{snap_num:03d}.hdf5"
                            model_run_name=f'run{run}_snap{snap_num}_galaxy{galaxy}'
                            obj = caesar.load(thisobj)
                            try:
                                    pos = np.array(obj.galaxies[galaxy].pos.in_units('code_length'))
                            except:
                                    print(f"run {run} has no galaxies pre snap {snap_num}")
                                    break
                            xpos, ypos, zpos = pos[0],pos[1],pos[2]
                            snap_redshift = obj.simulation.redshift
                            tcmb = 2.73*(1.+snap_redshift)
                            if snap_num in chosen_snaps:
                                    cmd = "./generate_model_files.sh "+str(model_dir_base)+' '+str(run)+' '+str(path)+' '+str(snap_num)+' '+str(model_run_name)+' '+str(galaxy)+' '+str(xpos)+' '+str(ypos)+' '+str(zpos)+' '+str(tcmb)
                                    call(cmd,shell=True)

        
