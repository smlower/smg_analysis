from __future__ import print_function
import matplotlib
matplotlib.use('agg') ## this calls matplotlib without a X-windows GUI
import numpy as np
import matplotlib.pyplot as plt
#import sys
#sys.path.insert(0,'/home/desika.narayanan/dope_viz/torrey_viz/torreylabtools/Python')
#import plotting.images.gas_images  as gas_images
import visualization.image_maker as pfh_image_maker
import os.path
import sys
import glob 
import caesar

data_dir = '/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run2_halo0_10myr/'
#galaxy_pos = [12166.10005621, 14948.90494145, 18212.95044001]
snap_list = glob.glob( data_dir +'/snapshot*.hdf5' )
n_frames = len(snap_list)

# First argument, if present, is resolution
if len(sys.argv) > 1:
    pixels          = int(sys.argv[1])
else:
    pixels = 256

# Second argument, if present, is the frame to render
if len(sys.argv) > 2:
    snapshot_to_render = int(sys.argv[2])
else:
    snapshot_to_render = np.arange(20,n_frames)
    #DEBUG
#    snapshot_to_render = np.arange(0,165)
print(snapshot_to_render)
y_resolution=pixels 	# typical resolutions are: 480, 720, 1080, etc.
aspect_ratio = 16.0/9.0  # ratio of xlen to ylen
pixels = y_resolution * aspect_ratio  # n_pixels, as needed by the code, sets the x pixel count.
tag=str(y_resolution)+'_16to9'



# trajectory parameters.  In this case, start at 1500 (code units = kpc), have a minimum dist of 150 (kpc), and do 3 rotations.
dmin     = 60
dmax     = 90
n_rotations = 0.25

# This sets how 'quickly' we go from dmax to dmin, and back.
c_for_traj = 2.0 * (dmax - dmin) / (1.0 * n_frames*n_frames)

for snapnr in [snapshot_to_render]:	
      obj = caesar.load(data_dir+f'/Groups/caesar_snapshot_{snapnr:03d}.hdf5')
      # the output file name
      #fname = '/blue/narayanan/s.lower/zoom_vis/vr_image_traj_'+tag+'_'+str(snapnr).zfill(4)+'.png'
      fname = '/home/s.lower/scripts/simba_smgs/vr_image_traj_'+tag+'_'+str(snapnr).zfill(4)+'.png'
      galaxy_pos = obj.halos[0].minpotpos.in_units('code_length').value
      # if the file already exists, it won't overwrite it.  If you want it overwritten, delete it.
      #snapnr_str = f'{snapnr:03d}'
      overwrite = True
      if overwrite == True:

	  # this stuff is not necessary!  
	  # But, if you do keep it, it changes the camera distance + viewing angle as a continuous function of
	  # the animation.  Obviously, you can mess with it based on preference and/or whatever.
          distance = dmin + c_for_traj * (snapnr - n_frames/2.0)**2
          angle_theta   = (1800.0*n_rotations) / (n_frames*1.0) * snapnr
          angle_theta1  = angle_theta + 3       
          angle_phi     = 60

          field_of_view = np.tan( 45.0 * np.pi / 180.0 ) * distance

          image,massmap = pfh_image_maker.image_maker( data_dir, snapnr,
                                                            snapdir_master='',
                                                            outdir_master='',
                                                            filename_set_manually='tmp_remove',
                                                            center_pos=galaxy_pos,
                                                            xrange=[-field_of_view * aspect_ratio, field_of_view * aspect_ratio],
                                                            yrange=[-field_of_view, field_of_view],
                                                            pixels=pixels,
                                                            project_to_camera=1,
                                                            h_rescale_factor=2.75,
                                                            phi = angle_phi, theta = angle_theta, include_lighting=0)


          # transpose the image.  This obviously dosen't matter if you don't care about orientation.
          new_image = np.zeros( (image.shape[1], image.shape[0], 3) )
          for ijk in range(3):
              new_image[:,:,ijk] = np.transpose( image[:,:,ijk] )

          # plot it.
          fig,ax=plt.subplots( figsize=(5.0*aspect_ratio ,5.0) )
          ax.imshow( new_image )
          ax.get_xaxis().set_visible(False)
          ax.get_yaxis().set_visible(False)
          fig.subplots_adjust( bottom=0, top=1, left=0, right=1)
          fig.savefig( fname , dpi=np.round(pixels/(5.0*aspect_ratio)) )
          plt.close()

