from hyperion.dust import IsotropicDust
import numpy as np
import ipdb,pdb
import astropy.units as u
import astropy.constants as constants


datafile = 'mw_rv31_li_draine.dat'

data = np.loadtxt(datafile,skiprows=3)
lam = data[:,0]*u.micron
albedo = data[:,1]
cos = data[:,2]
c_ext = data[:,3]
kappa = data[:,4]*u.cm**2./u.g
cos2 = data[:,5]

nu = (constants.c/lam).to(u.Hz)


#for some reason there's a few wavelengths that are fucked up and out of order
albedo = albedo[np.argsort(nu)]
cos = cos[np.argsort(nu)]
c_ext = c_ext[np.argsort(nu)]
kappa = kappa[np.argsort(nu)]
cos2 = cos2[np.argsort(nu)]
nu = nu[np.argsort(nu)]

#find any duplicates 9also present in the fuckups in this list)
unique_idx = np.unique(nu,return_index=True)[1]
albedo = albedo[unique_idx]
cos = cos[unique_idx]
c_ext = c_ext[unique_idx]
kappa = kappa[unique_idx]
cos2 = cos2[unique_idx]
nu = nu[unique_idx]

    


'''
#reverse all arrays to get in right order
nu = nu[::-1]
albedo = albedo[::-1]
cos = cos[::-1]
c_ext = c_ext[::-1]
kappa = kappa[::-1]
cos2 = cos2[::-1]
kappa = kappa[::-1]
'''

from hyperion.dust import IsotropicDust
d = IsotropicDust(nu.value,albedo,kappa.value)
d.write('mw_rv31_li_draine.hdf5')
