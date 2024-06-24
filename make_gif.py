import imageio
from glob import glob



filenames = []
for i in range(25):
    filenames.append(f'/home/s.lower/scripts/smg_movie/run5_halo0_snap{i:03d}.png')
    #filenames.append(f'/home/s.lower/scripts/smg_movie/run2_halo0_galaxy0_10myr_snap{i:03d}.png')
    #filenames.append(f'/blue/narayanan/s.lower/zoom_vis/interp_3myr_frame{i}.png')
    
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('/home/s.lower/scripts/smg_movie/run5_movie.gif', images, duration=0.2)
