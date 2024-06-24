# import SPHGR, numpy, pylab, and glob2
import yt
import yt.analysis_modules.sphgr.api as sphgr
import numpy as np
from glob2 import glob
import ipdb
from scipy.optimize import newton
from astropy.cosmology import Planck13
from astropy.io import ascii
from astropy.table import Table

#========================================
# naming schemes used to search for member files

#USER SETTABLE OPTIONS

BASEDIR = '/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run0_halo1/'
SNAPPRE = 'halo62.snap085.ml12.mufasa/output/'

#BASEDIR = '/Volumes/pegasus2/gadgetruns/'
#SNAPPRE='m12v_mr_Dec5_2013_3/'
LASTSNAP = 157 #only important for getting the snapshot numbers right,
               #but necessary for pd scripts
HALOS =True

FORCE_PROGEN_INDEX = False
FORCED_PROGEN_INDEX_VALUE = 0

#========================================



def get_sfr(obj,index):

    #ages
    simtime = Planck13.age(obj.redshift).value #what is the age of the Universe right now?
    
    galaxy_gas = obj.galhalo[INDEX].glist
    sindexes = obj.galhalo[INDEX].slist
        

    #needs to be fixed
    scalefactor = obj.particle_data['sage']
    formation_z = (1./scalefactor)-1.


    stellar_mass = obj.particle_data['smass'].in_units('Msun')
    
    #xz = newton_raphson(f,simtime,obj.redshift+0.5,tol=0.01)
    np.savez('junk.npz',simtime=simtime)
    
    #calculate the SFR for 50 Myr intervals
    xz_50 = newton(f2_50,obj.redshift+1)
    

    print '=========='
    print 'redshift is ',obj.redshift
    print 'newton_raphson derived redshift is for xz_50: ',xz_50
    print "newton_raphson derived delta t (for xz_50) = ",Planck13.age(obj.redshift).value-Planck13.age(xz_50).value

    w = np.where(formation_z <= xz_50)[0]
    sfr_50 = np.sum(stellar_mass[w])/50.e6



   
    return float(sfr_50.value)
    

def f2_50(formation_z):
    simtime = np.load('junk.npz')
    simtime = float(simtime['simtime'])

    print formation_z
    print (simtime-Planck13.age(formation_z).value)
    return (simtime-Planck13.age(formation_z).value)-0.05







#========================================
#MAIN CODE
#========================================


# query all available member files
MEMBERS = np.sort(glob('%s/%s/Groups/*.hdf5' % (BASEDIR,SNAPPRE)))

# set the galaxy of interest starting index
INDEX = 0

# create empty lists to hold Mstar and z
stellar_masses, gas_masses, h2_masses,hI_masses,halo_masses = [],[],[],[],[]
redshifts, hmr, fmr, instsfr, sfr_50,metallicity = [],[],[],[],[],[]
snapshotnames,snaps = [],[]
dm_masses = []
xpos,ypos,zpos = [],[],[]



# cycle through all MEMBERS in reverse
for i in reversed(range(1,len(MEMBERS))):

#for i in reversed(range(180,192))
    # load the current member file
    print 'loading %s' % MEMBERS[i]
    obj = sphgr.load_sphgr_data(MEMBERS[i])
    ds = yt.load(BASEDIR+SNAPPRE+obj.basename)
    obj.yt_dataset = ds


    
    snapshotnames.append(obj.basename)

    if HALOS == False:
        obj.galhalo = obj.galaxies
        halo_masses.append(-1)
    else:
        obj.galhalo = obj.halos
        halo_masses.append(obj.galhalo[INDEX].masses['dm'])
       
    # get the different masses
    stellar_masses.append(obj.galhalo[INDEX].masses['stellar'])
    gas_masses.append(obj.galhalo[INDEX].masses['gas'])
    h2_masses.append(obj.galhalo[INDEX].masses['H2'])
    hI_masses.append(obj.galhalo[INDEX].masses['HI'])

    
        

    redshifts.append(obj.redshift)
    metallicity.append(obj.galhalo[INDEX].metallicity)

    hmr.append(obj.galhalo[INDEX].radii['stellar_half_mass'])
    fmr.append(obj.galhalo[INDEX].radii['stellar'])

    instsfr.append(obj.galhalo[INDEX].sfr)
    sfr_50.append(get_sfr(obj,INDEX))

    xpos.append(obj.galhalo[INDEX].pos[0].in_units('code_length').value)
    ypos.append(obj.galhalo[INDEX].pos[1].in_units('code_length').value)
    zpos.append(obj.galhalo[INDEX].pos[2].in_units('code_length').value)

    snaps.append(LASTSNAP-(len(redshifts)-1))

    # set the new INDEX value to be used in the previous snapshot.  If
    # a -1 is encountered that means there was no progenitor and we
    # can break from the loop.
    
    if FORCE_PROGEN_INDEX == True: 
        INDEX = FORCED_PROGEN_INDEX_VALUE 
    else:
        INDEX = obj.galhalo[INDEX].progen_index
        if INDEX == -1:
            break


    phys_table = Table([snapshotnames[::-1],
                        snaps[::-1],
                        redshifts[::-1],
                        instsfr[::-1],
                        sfr_50[::-1],
                        stellar_masses[::-1],
                        gas_masses[::-1],
                        h2_masses[::-1],
                        hI_masses[::-1],
                        hmr[::-1],
                        fmr[::-1],
                        metallicity[::-1],
                        xpos[::-1],
                        ypos[::-1],
                        zpos[::-1],
                        halo_masses[::-1]],
                       names=['snapname',
                              'snap',
                              'redshift',
                              'instsfr',
                              'sfr_50',
                              'M*',
                              'Mgas',
                              'MH2',
                              'MHI',
                              'HMR',
                              'FMR',
                              'metallicity',
                              'xpos',
                              'ypos',
                              'zpos',
                              'mhalo'])
    
    
    if HALOS == False:
        outfile = BASEDIR+SNAPPRE+'/Groups/sphgr_physical_properties.dat'
        outsavefile = BASEDIR+SNAPPRE+'/Groups/sphgr_physical_properties.npz'
    else:
        outfile = BASEDIR+SNAPPRE+'/Groups/sphgr_physical_properties.halos.dat'
        outsavefile = BASEDIR+SNAPPRE+'/Groups/sphgr_physical_properties.halos.npz'


    ascii.write(phys_table,outfile)
    np.savez(outsavefile,snapshotnames=snapshotnames,
             snaps=snaps,
             redshifts=redshifts,
             instsfr=instsfr,
             sfr_50=sfr_50,
             stellar_masses=stellar_masses,
             gas_masses=gas_masses,
             h2_masses=h2_masses,
             hI_masses=hI_masses,
             halo_masses=halo_masses,
             hmr=hmr,
             fmr=fmr,
             metallicity=metallicity,
             xpos=xpos,
             ypos=ypos,
             zpos=zpos)
