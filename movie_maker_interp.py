import numpy as np
import util.utilities as util
import os.path
import visualization.image_maker as imaker
import gadget_lib.gadget as gadget
import caesar
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    "font.family": "STIXGeneral",
    "mathtext.fontset" : "cm"
})
##
## here is the main routine for the galaxy simulation movies: 
##   it builds the snapshot list, does the centering, goes through frames and calls
##   the appropriate routines for interpolation between snapshots, and then calls the 
##   image_maker routine to render the frame: many switches here (most don't need to be set), 
##   but tune it as you need to (I recommend a low snapshot spacing to test)
##


### Run time params #####

snapshot_dir='/orange/narayanan/s.lower/simba/m25n256_dm/zooms/output/run2_halo0_10myr/'
output_dir='/blue/narayanan/s.lower/zoom_vis/'
cosmo=1.
size=25000.
type_to_make='gas'
frames_per_gyr = 70.
i_snap_min = 60
i_snap_max = 61
set_fixed_center = 1

def movie_maker(\
    xmax_of_box=50., #"""scale (in code units) of movie box size"""
    snapdir='/n/scratch2/hernquist_lab/phopkins/zoom_test/from_scinet/hires_bh', #"""location of snapshots"""
    outputdir_master='/n/scratch2/hernquist_lab/phopkins/movie_frames/', #"""parent folder for frames dump"""
    show_gasstarxray='stars', #"""determines if image is 'stars','gas','xr' etc"""
    frames_per_gyr=100., #"""sets spacing of frames (even in time)"""
    cosmological=1, #"""is this is a cosmological (comoving) simulation?"""
    time_min=0, #"""initial time of movie"""
    time_max=0, #"""final time (if<=0, just set to time of maximum snapshot)"""
    frame_min=0, #"""initial frame number to process"""
    frame_max=1.0e10, #"""final frame number to process"""
    i_snap_min=0, #"""minimum snapshot number in list to scan"""
    i_snap_max=0, #"""maximum snapshot number to scan (if =0, will just use largest in snapdir)"""
    pixels=720, #"""images will be pixels*pixels"""
    show_time_label=0, #"""do or don't place time label on each frame"""
    show_scale_label=0, #"""do or don't place physical scale label on each frame"""
    temp_max=1.0e6, #"""maximum gas temperature for color-weighted gas images (as 'temp_cuts' in 3-color)"""
    temp_min=3.0e2, #"""minimum gas temperature for color-weighted gas images (as 'temp_cuts' in 3-color)"""
    theta_0=90., #"""(initial) polar angle (if rotating)"""
    phi_0=90., #"""(initial) azimuthal angle (if rotating)"""
    center_on_bh=0, #"""if have a bh particle, uses it for the centering"""
    center_on_com=0, #"""simple centering on center of mass"""
    set_fixed_center=0, #"""option to set a fixed center for the whole movie"""
    sdss_colors=0, #"""use sdss color-scheme for three-color images"""
    nasa_colors=1, #"""use nasa color-scheme for three-color images"""
    use_h0=1, #"""correct snapshots to physical units"""
    four_char=0, #"""snapshots have 4-character filenames"""
    skip_bh=0, #"""skips reading bh information in snapshots"""
    use_old_extinction_routine=0, #"""uses older (faster but less stable) stellar extinction routine"""
    add_extension='', #"""extension for movie snapshot images"""
    do_with_colors=1, #"""make color images"""
    threecolor=1, #"""do or dont use the standard three-color projection for images"""
    camera_opening_angle=45., #"""camera opening angle (specifies camera distance)"""
    scattered_fraction=0.01, #"""fraction of light that is unattenuated"""
    z_to_add=0.0, #"""add this metallicity to all particles"""
    min_stellar_age=0.0 #"""force star particles to be at least this old to avoid interpolation artifacts"""
    ):

    ## first decide whether its a gas or stellar image
    SHOW_STARS=0; SHOW_GAS=1;
    if((show_gasstarxray=='star') or (show_gasstarxray=='stars') or (show_gasstarxray=='st')): 
        SHOW_STARS=1; SHOW_GAS=0;

    ## parse the directory names (for file-naming conventions)
    #s0=snapdir.split("/"); snapdir_specific=s0[len(s0)-1]; n_s0=1;
    #if(len(snapdir_specific)<=1): snapdir_specific=s0[len(s0)-2]; n_s0=2;
    #snapdir_master=''; 
    #for j in s0[0:len(s0)+1-n_s0]: snapdir_master += str(j)+'/';
    #outputdir_master+='/' ## just to be safe
    snapdir_master = snapdir
    ## build the snapshot list in the directory of interest
    print('... building snapshot list in directory ...')
    snapshot_list = build_snapshot_list( snapdir, \
        use_h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh )
    if(i_snap_max<=0): i_snap_max=snapshot_list[snapshot_list.size-2];
    ## to avoid 'camera jitter', the safest thing (albeit more expensive) is to 
    ##   pre-compute the movie center at all snapshots, then use a smoothed tracking of it
    ##   -- call subroutine to check if its done for us, and if not, build it --
    print('... building/loading camera positions list ...')
    if set_fixed_center == 1:
        set_fixed_center = np.empty((len(snapshot_list)-1,3))
        for snap_id in snapshot_list[:-1]:
            print('... loading caesar catalogue to get galaxy pos...')
            obj = caesar.load(snapdir+f'/Groups/caesar_snapshot_{snap_id:03d}.hdf5')
            galaxy_pos = obj.halos[0].minpotpos.in_units('code_length').value
            set_fixed_center[np.where(snapshot_list == snap_id)[0],:] = galaxy_pos
        print(set_fixed_center)
        build_camera_centering( snapshot_list, snapdir, force_rebuild=1, \
        center_on_bh=center_on_bh,center_on_com=center_on_com,set_fixed_center=set_fixed_center,\
        use_h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh )
    else:
        build_camera_centering( snapshot_list, snapdir, force_rebuild=0, \
        center_on_bh=center_on_bh,center_on_com=center_on_com,set_fixed_center=set_fixed_center,\
        use_h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh )
        

    print('... building list of times for movie frames ...')
    ## now build the times for each frame of the movie
    time_frame_grid, a_scale_grid = build_time_frame_grid( snapdir, snapshot_list, \
          frames_per_gyr, time_min=time_min, time_max=time_max, \
          use_h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh)

    ##
    ## some (simple) defaults for the scaling of dynamic range, etc
    ##  (can do fancy things for movie panning/zooming etc by changing the variables here)
    ##
    t=0.0*time_frame_grid
    x_plot_scale_grid = t + xmax_of_box
    y_plot_scale_grid = x_plot_scale_grid
    #theta_frame_grid = t + 90. # should be in degrees
    theta_frame_grid = t + theta_0 # should be in degrees
    #phi_frame_grid = t + 90. ## x-y
    phi_frame_grid = t + phi_0 ## x-y
    dynrange_0 = 3.0e2
    maxden_0 = 3.8e7
    maxden_grid = t + maxden_0 * ((50./x_plot_scale_grid)**(0.3))
    if(SHOW_STARS==1):
        dynrange_0 *= 100.0 #XXX CCH: originally 3.0
        maxden_grid *= 100.0 #XXX CCH: originally 3.0
    maxden_grid /= 1.0e10 # recalling in gadget units, m=1.0e10 m_sun
    # layer below is for zoom_dw large-scale GAS sims ONLY
    #z_grid = 1./a_scale_grid - 1.
    #dynrange_0 *= 20. * (1.+np.exp(-(z_grid-2.)))
    #maxden_grid *= 3.
    dynrange_grid=dynrange_0 + 0.*x_plot_scale_grid

    
    print('... entering main snapshot loop ...')
    ## now enter main loop over snapshots in the simulation set
    snapshot_f_number_that_i_just_loaded = -1;
    snaps_to_do = (snapshot_list >= i_snap_min) & (snapshot_list <= i_snap_max)
    ii = np.arange(snapshot_list.size)
    snap_grid_to_loop = ii[snaps_to_do]
    for i_snap in snap_grid_to_loop:
        ## load header info for snapshot 'i'
        PPP_head = gadget.readsnap(snapdir,snapshot_list[i_snap],1,header_only=1,\
            h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh)
        if(cosmological==1):
            a_scale_i=PPP_head['time']; t_i=cosmological_time(a_scale_i); 
        else: 
            t_i=PPP_head['time']; a_scale_i=t_i
        a_scale_i=np.array(a_scale_i); t_i=np.array(t_i);
            
        ## load header info for snapshot 'f=i+1'
        PPP_head = gadget.readsnap(snapdir,snapshot_list[i_snap+1],1,header_only=1,\
            h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh)
        if(cosmological==1):
            a_scale_f=PPP_head['time']; t_f=cosmological_time(a_scale_f); 
        else: 
            t_f=PPP_head['time']; a_scale_f=t_f
        a_scale_f=np.array(a_scale_f); t_f=np.array(t_f);
        
        print('... snapshot headers loaded: getting timestep info: ...')
        ## now calculate whether there are any frames in the time-range between the snapshots
        delta_t = t_f - t_i
        print('Timestep is ti/tf/delta_t ',t_i,t_f,delta_t)
        i_frame = np.array(list(range(time_frame_grid.size)));
        check_frames_to_do = (time_frame_grid >= t_i) & (time_frame_grid <= t_f) & \
            (i_frame <= frame_max) & (i_frame >= frame_min);
        frames_to_do = i_frame[check_frames_to_do];
        n_frames_to_do = frames_to_do.size;

        if(n_frames_to_do>0):
            print('... processing ',n_frames_to_do,' frames in this snapshot interval ...')

            center_i = get_precalc_zoom_center(snapdir,a_scale_i,cosmological=cosmological)
            center_f = get_precalc_zoom_center(snapdir,a_scale_f,cosmological=cosmological)
            ## correct to comoving coordinates for interpolation (so correctly capture hubble flow)
            if (cosmological==1):
                center_i /= a_scale_i
                center_f /= a_scale_f
            print('... ... centered (i/f) at ',center_i,center_f)  

            print('... ... loading snapshot bricks ... ...')
            ## tolerance for keeping particles outside the box, for this calculation
            xmax_tmp=xmax_of_box*1.5 + 10.
            ## load the actual --data-- for snapshot (i): 
            id_i,m_i,x_i,y_i,z_i,vx_i,vy_i,vz_i,h_i,c_i,zm_i = \
                load_pos_vel_etc(snapdir, snapshot_list[i_snap],\
                    h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh, \
                    GAS=SHOW_GAS,BOX_MAX=xmax_tmp,CENTER_BOX=center_i, \
                    min_stellar_age=min_stellar_age);
            if (SHOW_STARS==1):
                id_i_g,m_i_g,x_i_g,y_i_g,z_i_g,vx_i_g,vy_i_g,vz_i_g,h_i_g,c_i_g,zm_i_g = \
                  load_pos_vel_etc(snapdir, snapshot_list[i_snap], \
                    h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh, \
                    GAS=1,SWAP_TEMP_RHO=1,BOX_MAX=xmax_tmp,CENTER_BOX=center_i, \
                    min_stellar_age=min_stellar_age);
            ## correct to comoving coordinates for interpolation (so correctly capture hubble flow)
            ## (put this here, splitting initial and final rescalings so dont do it twice on 'recycling' step)
            if (cosmological==1):
                for vec in [x_i,y_i,z_i,h_i,vx_i,vy_i,vz_i]: vec /= a_scale_i;
                if (SHOW_STARS==1):
                    for vec in [x_i_g,y_i_g,z_i_g,h_i_g,vx_i_g,vy_i_g,vz_i_g]: vec /= a_scale_i;

            ## now load snapshot (f) [should be new, have to load fresh] 
            id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,h_f,c_f,zm_f = \
                load_pos_vel_etc(snapdir, snapshot_list[i_snap+1], \
                    h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh, \
                    GAS=SHOW_GAS,BOX_MAX=xmax_tmp,CENTER_BOX=center_f);
            if (SHOW_STARS==1):
                id_f_g,m_f_g,x_f_g,y_f_g,z_f_g,vx_f_g,vy_f_g,vz_f_g,h_f_g,c_f_g,zm_f_g = \
                  load_pos_vel_etc(snapdir, snapshot_list[i_snap+1], \
                    h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh, \
                    GAS=1,SWAP_TEMP_RHO=1,BOX_MAX=xmax_tmp,CENTER_BOX=center_f);
            ## correct to comoving coordinates for interpolation (so correctly capture hubble flow)
            if (cosmological==1):
                for vec in [x_f,y_f,z_f,h_f,vx_f,vy_f,vz_f]: vec /= a_scale_f;
                if (SHOW_STARS==1):
                    for vec in [x_f_g,y_f_g,z_f_g,h_f_g,vx_f_g,vy_f_g,vz_f_g]: vec /= a_scale_f;
            snapshot_f_number_that_i_just_loaded = snapshot_list[i_snap+1];

            ## before going further, check that we actually have particles to process, 
            ##   and if not skip this frame
            if (m_i.size+m_f.size < 5):
                continue;
        
            print('... ... matching ids in the snapshots ... ...')
            ## predefine matches to save time in the loop below
            sub_id_i,sub_id_f,nomatch_from_i_in_f,nomatch_from_f_in_i = compile_matched_ids( id_i, id_f );
            if (SHOW_STARS==1):
                sub_id_i_g,sub_id_f_g,nomatch_from_i_in_f_g,nomatch_from_f_in_i_g = compile_matched_ids( id_i_g, id_f_g );
    
            print('... ... entering frame loop ... ...')
            ## loop over frames in between the two snapshots just pulled up
            for i_frame in range(n_frames_to_do):
                print(frames_to_do)
                print(i_frame)
                j_of_frame = frames_to_do[i_frame];
                dt = time_frame_grid[j_of_frame] - t_i
                time_of_frame = time_frame_grid[j_of_frame]

                print('... ... ... interpolating for frame ',i_frame+1,'/',n_frames_to_do,'... ... ...')
                ## this is the interpolation step between the two snapshots for each frame
                x_all,y_all,z_all,m_all,h_all,c_all,zm_all = interpolation_for_movie( dt, delta_t, \
                    m_i,h_i,c_i,zm_i, m_f,h_f,c_f,zm_f, \
                    x_i,y_i,z_i,vx_i,vy_i,vz_i, x_f,y_f,z_f,vx_f,vy_f,vz_f, \
                    sub_id_i,sub_id_f, nomatch_from_i_in_f,nomatch_from_f_in_i, \
                    center_i,center_f,use_polar_interpolation=0 )
                if (SHOW_STARS==1):
                    x_all_g,y_all_g,z_all_g,m_all_g,h_all_g,c_all_g,zm_all_g = \
                        interpolation_for_movie( dt, delta_t, \
                        m_i_g,h_i_g,c_i_g,zm_i_g,m_f_g,h_f_g,c_f_g,zm_f_g, \
                        x_i_g,y_i_g,z_i_g,vx_i_g,vy_i_g,vz_i_g, x_f_g,y_f_g,z_f_g,vx_f_g,vy_f_g,vz_f_g, \
                        sub_id_i_g,sub_id_f_g, nomatch_from_i_in_f_g,nomatch_from_f_in_i_g, \
                        center_i,center_f,use_polar_interpolation=0 )
                ## ok, now have the interpolated gas(+stellar) quantities needed for each image

                ## correct back from comoving coordinates for plotting (since units are physical)
                if (cosmological==1):
                    a = a_scale_grid[j_of_frame]
                    time_of_frame = a
                    for vec in [x_all,y_all,z_all,h_all]: vec *= a;
                    if (SHOW_STARS==1):
                        for vec in [x_all_g,y_all_g,z_all_g,h_all_g]: vec *= a;

                print('... ... ... centering and setting passed variables ... ... ...')
                ## re-center on the current frame center, interpolated between the snapshots
                cen = get_precalc_zoom_center(snapdir,a_scale_grid[j_of_frame],cosmological=cosmological)
                print('Center for frame j=',j_of_frame,' at ',cen)
                x_all-=cen[0]; y_all-=cen[1]; z_all-=cen[2];
                if (SHOW_STARS==1):
                    x_all_g-=cen[0]; y_all_g-=cen[1]; z_all_g-=cen[2];
        
    
                ## set dynamic ranges of image
                maxden=maxden_grid[j_of_frame]
                dynrange=dynrange_grid[j_of_frame]
                print(f'dynamic range: {dynrange}')
                ## set image spatial scale 
                xr_0=x_plot_scale_grid[j_of_frame]
                yr_0=y_plot_scale_grid[j_of_frame]
                xr=[-xr_0,xr_0]; yr=[-yr_0,yr_0]; zr_0=max([xr_0,yr_0]); zr=[-zr_0,zr_0]
                ## set viewing angle for image
                theta = theta_frame_grid[j_of_frame]
                phi   = phi_frame_grid[j_of_frame]

                ## some final variable cleaning/prep before imaging routine:
                if (SHOW_STARS==0):
                    gdum=np.array([0]);
                    x_all_g=y_all_g=z_all_g=m_all_g=h_all_g=c_all_g=zm_all_g=gdum;
                else:
                    gdum=np.zeros(m_all_g.size)+1.;
                        
                print('... ... ... sending to main imaging routine ... ... ...')

                print(f'x_all = {x_all}')
                ## alright, now we can actually send this to the imaging routine 
                ##   to make the frame image! 
                #print(f'snapdir_specific: {snapdir_specific}')
                #print(f'snapdir_master: {snapdir_master}')
                image24, massmap = \
                  imaker.image_maker( '', i_snap,#j_of_frame, \
                    snapdir_master=snapdir, outdir_master=outputdir_master, 
                    theta=theta, phi=phi, dynrange=dynrange, maxden=maxden, 
                    show_gasstarxray=show_gasstarxray, add_extension=add_extension, \
                    show_time_label=show_time_label, show_scale_label=show_scale_label, \
                    use_h0=use_h0, cosmo=cosmological, coordinates_cylindrical=0, \
                    set_percent_maxden=0, set_percent_minden=0, pixels=pixels, \
                    center_on_com=0, center_on_bh=0, center=[0., 0., 0.], \
                    xrange=xr, yrange=yr, zrange=zr, 
                    project_to_camera=1, camera_opening_angle=camera_opening_angle, \
                    center_is_camera_position=0, camera_direction=[0.,0.,-1.], \
                    nasa_colors=nasa_colors, sdss_colors=sdss_colors, \
                    use_old_extinction_routine=use_old_extinction_routine, \
                    do_with_colors=do_with_colors, threecolor=threecolor, \
                    log_temp_wt=1, set_temp_min=temp_min, set_temp_max=temp_max,
                    gas_map_temperature_cuts=np.array([temp_min, temp_max]), 
                    input_data_is_sent_directly=1, include_lighting=SHOW_GAS, \
                    m_all=m_all,x_all=x_all,y_all=y_all,z_all=z_all,c_all=c_all,h_all=h_all,zm_all=zm_all+z_to_add,\
                    gas_u=gdum,gas_rho=c_all_g,gas_numh=gdum,gas_nume=gdum,gas_hsml=h_all_g,gas_metallicity=zm_all_g+z_to_add,\
                    gas_mass=m_all_g,gas_x=x_all_g,gas_y=y_all_g,gas_z=z_all_g,time=time_of_frame, \
                    scattered_fraction=scattered_fraction,min_stellar_age=min_stellar_age );
                ## don't need to do anything else here, the image & datafile are dumped!
                print('... ... ... frame ',j_of_frame,'complete! ... ... ...')
                
                fig,ax=plt.subplots( figsize=(5.0*(16./9.),5.0) )
                ax.imshow(image24)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                fig.subplots_adjust( bottom=0, top=1, left=0, right=1)
                plt.savefig(outputdir_master+f'interp_image_snap{i_snap}_frame{j_of_frame}.png',dpi=200)
    return 1; ## success!



def ok_scan(input,xmax=1.0e10,pos=0):
    if (pos==1):
        return (np.isnan(input)==False) & (np.abs(input)<=xmax) & (input > 0.);
    else:
        return (np.isnan(input)==False) & (np.abs(input)<=xmax);


def idmatch(id_i, id_f): 
    ## match items in set (i) to those in set (f)
    ##  --- this is particularly speedy when the ids are all unique (case here) --
    index_dict_i = dict((k,i) for i,k in enumerate(id_i));
    index_dict_f = dict((k,i) for i,k in enumerate(id_f));
    inter = set(id_i).intersection(set(id_f));
    indices_i = np.array([ index_dict_i[x] for x in inter ]);
    indices_f = np.array([ index_dict_f[x] for x in inter ]);
    return indices_i, indices_f;


def compile_matched_ids( id_i, id_f ):
    sub_id_i, sub_id_f = idmatch(id_i,id_f)
    
    nomatch_from_i_in_f = (id_i > -1.0e40) ## should return all true
    if (sub_id_i.size > 0):
        nomatch_from_i_in_f[sub_id_i] = False ## is matched
    
    nomatch_from_f_in_i = (id_f > -1.0e40) 
    if (sub_id_f.size > 0):
        nomatch_from_f_in_i[sub_id_f] = False
    
    return sub_id_i, sub_id_f, nomatch_from_i_in_f, nomatch_from_f_in_i


def interpolation_for_movie_pos( dt, delta_t, u_i,u_f, vu_i,vu_f, \
    sub_id_i,sub_id_f, nomatch_from_i_in_f,nomatch_from_f_in_i , periodic=0, order=3 ):

    dt=np.array(dt); delta_t=np.array(delta_t); ## make sure casting is ok
    tau = dt/delta_t;
    Nmatched=u_i[sub_id_i].size;
    Nunmatched_i=u_i[nomatch_from_i_in_f].size;
    Nunmatched_f=u_f[nomatch_from_f_in_i].size;
    u=np.zeros(Nmatched+Nunmatched_i+Nunmatched_f);

    # first compute for objects in (i) with a match in (f)
    if (Nmatched>0):
        xi = np.copy(u_i[sub_id_i])  ; xf = np.copy(u_f[sub_id_f])  ;
        vi = np.copy(vu_i[sub_id_i]) ; vf = np.copy(vu_f[sub_id_f]) ;
        if (periodic==1):
            ## u is periodic in 2pi, so estimate its final position and wrap coordinate
            ## guess where its going to be to get 'best estimate' of the number of revolutions
            #x_exp = xi + 0.5*(vf+vi)*delta_t # actually can get some errors when vf/i switch signs...
            x_exp = xi + vi*delta_t
            ## now pin the final location to the appropriate value
            xf = get_closest_periodic_value( x_exp, xf , xi)
        
        if (order==3):
            x2 = 3.*(xf-xi) - (2.*vi+vf)*delta_t
            x3 = -2.*(xf-xi) + (vi+vf)*delta_t
            ## third-order interpolation: enables exact matching of x, v, but can 'overshoot'
            u[0:Nmatched] = xi + vi*dt + x2*tau*tau + x3*tau*tau*tau
        elif (order==2):
            ## second-order: minimizes absolute velocity difference (more stable, no exact velocity matching)
            x2 = (vf-vi)*delta_t/2.
            x1 = (xf-xi) - x2
            u[0:Nmatched] = xi + x1*tau + x2*tau*tau
        else:
            ## use linear interpolation below if for some reason this is unstable: least accurate, but most stable
            u[0:Nmatched] = u_i[sub_id_i] + (u_f[sub_id_f]-u_i[sub_id_i])*tau 
        
    ## now unmatched: first those in (i) with no match in (f)
    if(Nunmatched_i>0):
        u[Nmatched:Nmatched+Nunmatched_i] = \
            u_i[nomatch_from_i_in_f]+vu_i[nomatch_from_i_in_f]*dt;
	## now unmatched: now those in (f) with no match in (i)
    if(Nunmatched_f>0):
        u[Nmatched+Nunmatched_i:Nmatched+Nunmatched_i+Nunmatched_f] = \
            u_f[nomatch_from_f_in_i]+vu_f[nomatch_from_f_in_i]*(dt-delta_t);

    return u;


def interpolation_for_movie_jhat( dt, delta_t, j_i,j_f, \
    sub_id_i,sub_id_f, nomatch_from_i_in_f,nomatch_from_f_in_i ):

    dt=np.array(dt); delta_t=np.array(delta_t); ## make sure casting is ok
    tau = dt/delta_t;
    u_i = np.zeros(j_i[:,0].size)
    u_f = np.zeros(j_f[:,0].size)
    Nmatched=u_i[sub_id_i].size;
    Nunmatched_i=u_i[nomatch_from_i_in_f].size;
    Nunmatched_f=u_f[nomatch_from_f_in_i].size;
    j_hat=np.zeros((Nmatched+Nunmatched_i+Nunmatched_f,3));

    # first compute for objects in (i) with a match in (f)
    if (Nmatched>0):
        ## initialize matrices
        pt_i = np.zeros(Nmatched); pt_f = 0.*pt_i; 
        p_i = np.zeros((Nmatched,3)); p_f = 0.*p_i; v_f = 0.*p_f
        for j in [0,1,2]:
            p_i[:,j] = j_i[sub_id_i,j]
            p_f[:,j] = j_f[sub_id_f,j]
            pt_i += p_i[:,j]*p_i[:,j]
            pt_f += p_f[:,j]*p_f[:,j]
        ## normalize vectors to unity
        pt_i = np.sqrt(pt_i); pt_f = np.sqrt(pt_f);
        for j in [0,1,2]:
            p_i[:,j] /= pt_i
            p_f[:,j] /= pt_f
            
        ## build a rotation matrix between the j_hat values at two different timesteps:
        cos_rot_ang = p_i[:,0]*p_f[:,0] + p_i[:,1]*p_f[:,1] + p_i[:,2]*p_f[:,2]
        angle = np.arccos( cos_rot_ang )

        angle_to_rotate = angle * tau 
        cos_ang = np.cos(angle_to_rotate)
        sin_ang = np.sin(angle_to_rotate)

        ## build perpendicular vectors and normalize
        k_rot = cross( p_i, p_f )
        kk=0.*pt_i;
        for j in [0,1,2]: kk += k_rot[:,j]*k_rot[:,j]
        kk = np.sqrt(kk)
        for j in [0,1,2]: k_rot[:,j] /= kk
        
        k_cross_p = cross( k_rot, p_i )
        kk=0.*pt_i;
        for j in [0,1,2]: kk += k_cross_p[:,j]*k_cross_p[:,j]
        kk = np.sqrt(kk)
        for j in [0,1,2]: k_cross_p[:,j] /= kk
        
        k_dot_p = k_rot[:,0]*p_i[:,0] + k_rot[:,1]*p_i[:,1] + k_rot[:,2]*p_i[:,2]
        for j in [0,1,2]:
            v_f[:,j] = p_i[:,j]*cos_ang + k_cross_p[:,j]*sin_ang + k_rot[:,j]*k_dot_p*(1.-cos_ang)

        for j in [0,1,2]:
            j_hat[0:Nmatched,j] = v_f[:,j]  ## now have the j_hat vector for this time

    ## now unmatched: first those in (i) with no match in (f)
    if(Nunmatched_i>0):
        for j in [0,1,2]:
            j_hat[Nmatched:Nmatched+Nunmatched_i,j] = j_i[nomatch_from_i_in_f,j]

	## now unmatched: now those in (f) with no match in (i)
    if(Nunmatched_f>0):
        for j in [0,1,2]:
            j_hat[Nmatched+Nunmatched_i:Nmatched+Nunmatched_i+Nunmatched_f] = j_f[nomatch_from_f_in_i]

    return j_hat;



def interpolation_for_movie_scalar(dt, delta_t, u_i,u_f, \
    sub_id_i,sub_id_f, nomatch_from_i_in_f,nomatch_from_f_in_i, \
    mass_rescale=0, hsml_rescale=0, age_rescale=0 ):

    dt=np.array(dt); delta_t=np.array(delta_t); ## make sure casting is ok
    tau = dt/delta_t;
    Nmatched=u_i[sub_id_i].size;
    Nunmatched_i=u_i[nomatch_from_i_in_f].size;
    Nunmatched_f=u_f[nomatch_from_f_in_i].size;
    u=np.zeros(Nmatched+Nunmatched_i+Nunmatched_f);
    
    # first compute for objects in (i) with a match in (f)
    if (Nmatched>0):
        u[0:Nmatched] = u_i[sub_id_i] + (u_f[sub_id_f]-u_i[sub_id_i])*tau
        if(age_rescale==1):
            print('RAMBO THAT SHIT')
            #for ii in range(Nmatched):
            #    print 'a: ',dt, delta_t, u_i[sub_id_i[ii]], u_f[sub_id_f[ii]], u[ii]
            #u[0:Nmatched] = u_i[sub_id_i]
    ## now unmatched: first those in (i) with no match in (f)
    if(Nunmatched_i>0):
        if(mass_rescale==1):
            u_t = u_i[nomatch_from_i_in_f] * (1.-tau**(1./5.))**5.
        elif(hsml_rescale==1):
            u_t = u_i[nomatch_from_i_in_f] / ((1.-tau**(1./3.))**3.+1.0e-5)
        else:
            u_t = u_i[nomatch_from_i_in_f]
        u[Nmatched:Nmatched+Nunmatched_i] = u_t;
	## now unmatched: now those in (f) with no match in (i)
    if(Nunmatched_f>0):
        if(mass_rescale==1):
            u_t = u_f[nomatch_from_f_in_i] * (1.-(1.-tau)**(1./5.))**5.
        elif(hsml_rescale==1):
            u_t = u_f[nomatch_from_f_in_i] / ((1.-(1.-tau)**(1./3.))**3.+1.0e-5)
        else:
            u_t = u_f[nomatch_from_f_in_i]
        u[Nmatched+Nunmatched_i:Nmatched+Nunmatched_i+Nunmatched_f] = u_t;

    return u;


def cosmological_time(a,h=0.71,Omega_M=0.27):
    ## exact solution for a flat universe
    x=Omega_M/(1.-Omega_M) / (a*a*a);
    t=(2./(3.*np.sqrt(1.-Omega_M))) * np.log( np.sqrt(x) / (-1. + np.sqrt(1.+x)) );
    t *= 13.777 * (0.71/h); ## in Gyr
    return t;
    
    
def cross(x,y):
    #return np.cross(x,y,axis=1)
    c=0.*x
    c[:,0] = x[:,1]*y[:,2] - x[:,2]*y[:,1]
    c[:,1] =-x[:,0]*y[:,2] + x[:,2]*y[:,0]
    c[:,2] = x[:,0]*y[:,1] - x[:,1]*y[:,0]
    return c


def xyz_to_polar(x,y,z,vx,vy,vz):
    for q in [x,y,z,vx,vy,vz]: q=np.array(q,dtype='d')+1.0e-10
    x += 1.0e-5; vy += 1.0e-5;

    ## get the angular momentum vector (to use for interpolation)
    r = np.sqrt( x*x + y*y + z*z )
    pos = np.zeros([x.size,3],dtype='d'); 
    pos[:,0]=x/r; pos[:,1]=y/r; pos[:,2]=z/r;
    v = np.sqrt( vx*vx + vy*vy + vz*vz )
    vel = np.zeros([vx.size,3],dtype='d'); 
    vel[:,0]=vx/v; vel[:,1]=vy/v; vel[:,2]=vz/v;
    
    j_hat = cross( pos, vel )
    jj = np.sqrt( j_hat[:,0]*j_hat[:,0] + j_hat[:,1]*j_hat[:,1] + j_hat[:,2]*j_hat[:,2] )
    for j in [0,1,2]: j_hat[:,j] /= jj
    
    ## get spherical polar coordinates
    v_r = vx*pos[:,0] + vy*pos[:,1] + vz*pos[:,2]

    ## now get the vector perpendicular to both j and r
    phi_hat = cross( j_hat, pos )
    jj = np.sqrt( phi_hat[:,0]*phi_hat[:,0] + phi_hat[:,1]*phi_hat[:,1] + phi_hat[:,2]*phi_hat[:,2] )
    for j in [0,1,2]: phi_hat[:,j] /= jj

    v_phi = vx*phi_hat[:,0] + vy*phi_hat[:,1] + vz*phi_hat[:,2]
    v_phi /= r

    ## define the absolute 'phi' relative to an arbitrary (but fixed) 90-degree rotation 
    ##   of the angular momentum vector j
    x_jhat = 0.*j_hat
    x_jhat[:,0]=0.*j_hat[:,0]; x_jhat[:,1]=-j_hat[:,2]; x_jhat[:,2]=j_hat[:,1]
    jj = np.sqrt( x_jhat[:,0]*x_jhat[:,0] + x_jhat[:,1]*x_jhat[:,1] + x_jhat[:,2]*x_jhat[:,2] )
    for j in [0,1,2]: x_jhat[:,j] /= jj
    
    ## generate y-vector by cross-product a x b = c (gaurantees right-hand rule)
    y_jhat = cross( j_hat, x_jhat )
    jj = np.sqrt( y_jhat[:,0]*y_jhat[:,0] + y_jhat[:,1]*y_jhat[:,1] + y_jhat[:,2]*y_jhat[:,2] )
    for j in [0,1,2]: y_jhat[:,j] /= jj

    ## now project r onto this, to obtain the components in this plane
    x_for_phi = x_jhat[:,0]*pos[:,0] + x_jhat[:,1]*pos[:,1] + x_jhat[:,2]*pos[:,2]
    y_for_phi = y_jhat[:,0]*pos[:,0] + y_jhat[:,1]*pos[:,1] + y_jhat[:,2]*pos[:,2]
    phi = np.arctan2( y_for_phi , x_for_phi )
    ## reset to a 0-2pi system instead of numpy's -pi,pi system
    lo = (phi < 0.); phi[lo] += 2.*np.pi

    return r, v_r, phi, v_phi, j_hat

    
## simple function to wrap x/y/z coordinates near the box edges, to give 
##   the 'correct' x-x0 in periodic coordinates
def periodic_dx( x, x0, box ):
    b0 = box / 2.
    dx = x - x0
    too_large = (dx > b0)
    dx[too_large] -= box
    too_small = (dx < -b0)
    dx[too_small] += box
    return dx
    
    
def get_closest_periodic_value( x_expected, x_final, x_initial, limiter=1.0 ):
    x_remainder = np.mod(x_expected, 2.*np.pi)
    dx = periodic_dx( x_final, x_remainder, 2.*np.pi ) ## returns wrapped dx from x_remainder
    dx_full = x_expected + dx - x_initial
    
    return dx_full + x_initial
    
    # now put a limiter so we don't allow some crazy number of 'laps':
    sign_dx = 0.*dx_full + 1.
    sign_dx[dx_full<0.]=-1.
    dx_full_abs = (sign_dx*dx_full) 
    dx_full_cycles = dx_full_abs / (2.*np.pi) # number of 'laps'
    dx_full_remainder = np.mod(dx_full_cycles, 1.0)
    
    dx_corrected = np.copy(dx_full)
    lminusone = limiter - 1.
    hi = (dx_full_cycles > limiter) & (dx_full_remainder > lminusone)
    dx_corrected[hi] = 2.*np.pi*sign_dx[hi]*dx_full_remainder[hi]
    hi = (dx_full_cycles > limiter) & (dx_full_remainder < lminusone)
    dx_corrected[hi] = 2.*np.pi*sign_dx[hi]*(1.+dx_full_remainder[hi])
    
    return dx_corrected + x_initial


def interpolation_for_movie(dt, delta_t, \
	    m_i,h_i,c_i,zm_i,\
	    m_f,h_f,c_f,zm_f, \
	    x_in,y_in,z_in,vx_in,vy_in,vz_in, \
	    x_fn,y_fn,z_fn,vx_fn,vy_fn,vz_fn, \
	    s_i,s_f,nm_i_f,nm_f_i, \
	    cen_i,cen_f , use_polar_interpolation=1 ):
    
    x_i = np.copy(x_in) - cen_i[0]
    y_i = np.copy(y_in) - cen_i[1]
    z_i = np.copy(z_in) - cen_i[2]
    vx_i = np.copy(vx_in) - (cen_f[0]-cen_i[0])/delta_t
    vy_i = np.copy(vy_in) - (cen_f[1]-cen_i[1])/delta_t
    vz_i = np.copy(vz_in) - (cen_f[2]-cen_i[2])/delta_t
    x_f = np.copy(x_fn) - cen_f[0]
    y_f = np.copy(y_fn) - cen_f[1]
    z_f = np.copy(z_fn) - cen_f[2]
    vx_f = np.copy(vx_fn) - (cen_f[0]-cen_i[0])/delta_t
    vy_f = np.copy(vy_fn) - (cen_f[1]-cen_i[1])/delta_t
    vz_f = np.copy(vz_fn) - (cen_f[2]-cen_i[2])/delta_t
    
    #for ui,uin,j in zip([x_i,y_i,z_i],[x_in,y_in,z_in],[0,1,2]): ui = uin - cen_i[j]
    #for uf,ufn,j in zip([x_f,y_f,z_f],[x_fn,y_fn,z_fn],[0,1,2]): uf = ufn - cen_f[j]
    #for vui,vuin,j in zip([vx_i,vy_i,vz_i],[vx_in,vy_in,vz_in],[0,1,2]): vui = vuin - (cen_f[j]-cen_i[j])/delta_t
    #for vuf,vufn,j in zip([vx_f,vy_f,vz_f],[vx_fn,vy_fn,vz_fn],[0,1,2]): vuf = vufn - (cen_f[j]-cen_i[j])/delta_t

    if (use_polar_interpolation==1):
        r_i,vr_i,phi_i,vphi_i,j_hat_i = xyz_to_polar(x_i,y_i,z_i,vx_i,vy_i,vz_i)
        r_f,vr_f,phi_f,vphi_f,j_hat_f = xyz_to_polar(x_f,y_f,z_f,vx_f,vy_f,vz_f)

        # r gets treated like a normal spatial coordinate (except cannot go <0; ignore for now?)
        r = interpolation_for_movie_pos(dt,delta_t, r_i,r_f,vr_i,vr_f, s_i,s_f,nm_i_f,nm_f_i,order=1)
        # predict the 'final' phi, theta, knowing that these are actually periodic:
        # note for phi, need third-order: inexact matching can leave 'wedges' missing from orbits
        phi = interpolation_for_movie_pos(dt,delta_t, phi_i,phi_f,vphi_i,vphi_f, s_i,s_f,nm_i_f,nm_f_i, periodic=1,order=3)
        x_t = r * np.cos(phi);  y_t = r * np.sin(phi);

        j_hat = interpolation_for_movie_jhat(dt,delta_t, j_hat_i,j_hat_f, s_i,s_f,nm_i_f,nm_f_i )
        ## define the absolute 'phi' relative to an arbitrary (but fixed) 90-degree rotation of the angular momentum vector j
        x_jhat = 0.*j_hat
        x_jhat[:,0]=0.*j_hat[:,0]; x_jhat[:,1]=-j_hat[:,2]; x_jhat[:,2]=j_hat[:,1]
        jj = np.sqrt( x_jhat[:,0]*x_jhat[:,0] + x_jhat[:,1]*x_jhat[:,1] + x_jhat[:,2]*x_jhat[:,2] )
        for j in [0,1,2]: x_jhat[:,j] /= jj
        ## generate y-vector by cross-product a x b = c (gaurantees right-hand rule)
        y_jhat = cross( j_hat, x_jhat )
        jj = np.sqrt( y_jhat[:,0]*y_jhat[:,0] + y_jhat[:,1]*y_jhat[:,1] + y_jhat[:,2]*y_jhat[:,2] )
        for j in [0,1,2]: y_jhat[:,j] /= jj

        x = x_t*x_jhat[:,0] + y_t*y_jhat[:,0]
        y = x_t*x_jhat[:,1] + y_t*y_jhat[:,1]
        z = x_t*x_jhat[:,2] + y_t*y_jhat[:,2]

        ## use cartesian interpolation for some points
        x_c=interpolation_for_movie_pos(dt,delta_t, x_i,x_f,vx_i,vx_f, s_i,s_f,nm_i_f,nm_f_i,order=1)
        y_c=interpolation_for_movie_pos(dt,delta_t, y_i,y_f,vy_i,vy_f, s_i,s_f,nm_i_f,nm_f_i,order=1)
        z_c=interpolation_for_movie_pos(dt,delta_t, z_i,z_f,vz_i,vz_f, s_i,s_f,nm_i_f,nm_f_i,order=1)

        ## combine velocities into initial and final matched vectors
        Nmatched=x_i[s_i].size;
        Nunmatched_i=x_i[nm_i_f].size;
        Nunmatched_f=x_f[nm_f_i].size;
        u=np.zeros(Nmatched+Nunmatched_i+Nunmatched_f);
        vr=0.*u; vphi=0.*u; rr=0.*u; rvphi_i=r_i*vphi_i; rvphi_f=r_f*vphi_f;
        if (Nmatched>0):
            vr[0:Nmatched] = np.sqrt((vr_i[s_i]**2.+vr_f[s_f]**2.)/2.)
            vphi[0:Nmatched] = np.sqrt((rvphi_i[s_i]**2.+rvphi_f[s_f]**2.)/2.)
            rr[0:Nmatched] = np.sqrt((r_i[s_i]**2.+r_f[s_f]**2.)/2.)
        if(Nunmatched_i>0):
            vr[Nmatched:Nmatched+Nunmatched_i] = vr_i[nm_i_f]
            vphi[Nmatched:Nmatched+Nunmatched_i] = rvphi_i[nm_i_f]
            rr[Nmatched:Nmatched+Nunmatched_i] = r_i[nm_i_f]
        if(Nunmatched_f>0):
            vr[Nmatched+Nunmatched_i:Nmatched+Nunmatched_i+Nunmatched_f] = vr_f[nm_f_i]
            vphi[Nmatched+Nunmatched_i:Nmatched+Nunmatched_i+Nunmatched_f] = rvphi_f[nm_f_i]
            rr[Nmatched+Nunmatched_i:Nmatched+Nunmatched_i+Nunmatched_f] = r_f[nm_f_i]
        ## attempt to guess what's 'rotation supported' and not, and use appropriate interpolation
        vr2 = vr*vr
        v2 = vr2 + vphi*vphi
        #no_rot_support = ( vr2 > v2/3. ) | ( rr > 5. ) # usually does ok, too much 'breathing' at late times
        no_rot_support = ( vr2 > v2/2. ) | ( rr > 30. ) 
        x[no_rot_support] = x_c[no_rot_support]
        y[no_rot_support] = y_c[no_rot_support]
        z[no_rot_support] = z_c[no_rot_support]

    else:
        x=interpolation_for_movie_pos(dt,delta_t, x_i,x_f,vx_i,vx_f, s_i,s_f,nm_i_f,nm_f_i)
        y=interpolation_for_movie_pos(dt,delta_t, y_i,y_f,vy_i,vy_f, s_i,s_f,nm_i_f,nm_f_i)
        z=interpolation_for_movie_pos(dt,delta_t, z_i,z_f,vz_i,vz_f, s_i,s_f,nm_i_f,nm_f_i)

    x = x + (cen_f[0]-cen_i[0]) * dt / delta_t + cen_i[0]
    y = y + (cen_f[1]-cen_i[1]) * dt / delta_t + cen_i[1]
    z = z + (cen_f[2]-cen_i[2]) * dt / delta_t + cen_i[2]

    m=interpolation_for_movie_scalar(dt,delta_t, m_i,m_f, s_i,s_f,nm_i_f,nm_f_i, mass_rescale=1)
    h=interpolation_for_movie_scalar(dt,delta_t, h_i,h_f, s_i,s_f,nm_i_f,nm_f_i, hsml_rescale=1)
    c=interpolation_for_movie_scalar(dt,delta_t, c_i,c_f, s_i,s_f,nm_i_f,nm_f_i, age_rescale=1)
    #c=np.log10(interpolation_for_movie_scalar(dt,delta_t, 10.**c_i,10.**c_f, s_i,s_f,nm_i_f,nm_f_i))
    zm=interpolation_for_movie_scalar(dt,delta_t, zm_i,zm_f, s_i,s_f,nm_i_f,nm_f_i)

    return x,y,z,m,h,c,zm;


## here's the grunt work of loading the data we'll need 
def load_pos_vel_etc( snapdir, snapnum, \
        h0=1, four_char=0, cosmological=0, skip_bh=0, \
        use_rundir=1, #""" make sure to change this as appropriate for the system!"""
        GAS=0, SWAP_TEMP_RHO=0, BOX_MAX=0, CENTER_BOX=[0.,0.,0.], min_stellar_age=0. ):

    have=0; have_h_stars=0;
    ptypes=[0] ## just gas
    if(GAS==0): 
        if (cosmological==1): 
            ptypes=[4] ## just new stars
        else:
            ptypes=[2,3,4] ## bulge+disk+new stars

    for ptype in ptypes:
        ppp=gadget.readsnap(snapdir,snapnum,ptype,h0=h0,cosmological=cosmological,skip_bh=skip_bh);
        ppp_head=gadget.readsnap(snapdir,snapnum,ptype,header_only=1,h0=h0,cosmological=cosmological,skip_bh=skip_bh);
        if(ppp['k']==1):
            n=ppp['m'].size;
            if(n>1):
                m=ppp['m']; id=ppp['id'];
                p=ppp['p']; x=p[:,0]; y=p[:,1]; z=p[:,2]; 
                v=ppp['v']; vx=v[:,0]; vy=v[:,1]; vz=v[:,2]; 
                if (ptype==0): 
                    h=ppp['h']; zm=ppp['z']; 
                    if(len(zm.shape)>1): zm=zm[:,0]
                    if(SWAP_TEMP_RHO==1):
                        cc=ppp['rho']
                        cc_temp=gadget.gas_temperature(ppp['u'],ppp['ne']);
                        zm[cc_temp>1.0e6]=1.0e-6; ## don't allow hot gas to have dust
                    else:
                        cc=gadget.gas_temperature(ppp['u'],ppp['ne']);
                        #cc[cc<50.]=50.; ## purely to prevent very low values making the image bad
                if (ptype==4):
                    cc=gadget.get_stellar_ages(ppp,ppp_head,cosmological=cosmological);
                    cc=np.maximum(cc,min_stellar_age)
                    zm=ppp['z']; 
                    if(len(zm.shape)>1): zm=zm[:,0]
                if (ptype==2): ## need to assign ages and metallicities
                    # bad! this was causing bizzarre interpolations, because its re-initializing for each snapshot
                    #cc=np.random.rand(n)*(ppp_head['time']+4.0);
                    #cc=np.maximum(cc,min_stellar_age)
                    #zm=(np.random.rand(n)*(1.0-0.1)+0.1) * 0.02;
                    # use a constant formation time for these particles
                    cc = np.maximum(min_stellar_age , np.zeros(n)+4.0+ppp_head['time'])
                    zm = np.zeros(n) + 0.01;
                    # or allow a distribution, but one anchored to IDs so it behaves properly across snapshots
                    idx = (1.*(ppp['id']-np.min(ppp['id']))) / (1.*(np.max(ppp['id'])-np.min(ppp['id'])));
                    cc = np.maximum(min_stellar_age , ppp_head['time'] + 4.0*idx);
                    zm = 0.02 * (0.1 + (1.-0.1)*idx);
                if (ptype==3): ## need to assign ages and metallicities
                    # bad! this was causing bizzarre interpolations, because its re-initializing for each snapshot
                    #cc=np.random.rand(n)*(ppp_head['time']+12.0);
                    #cc=np.maximum(cc,min_stellar_age)
                    #zm=(np.random.rand(n)*(0.3-0.03)+0.03) * 0.02;
                    # use a constant formation time for these particles
                    cc = np.maximum(min_stellar_age , np.zeros(n)+12.0+ppp_head['time'])
                    zm = np.zeros(n) + 0.002;
                    # or allow a distribution, but one anchored to IDs so it behaves properly across snapshots
                    idx = (1.*(ppp['id']-np.min(ppp['id']))) / (1.*(np.max(ppp['id'])-np.min(ppp['id'])));
                    cc = np.maximum(min_stellar_age , ppp_head['time'] + 12.0*idx);
                    zm = 0.02 * (0.03 + (0.3-0.03)*idx);
                hstars_should_concat=0;
                if((ptype>0) & (have_h_stars==0)):
                    h=gadget.load_allstars_hsml(snapdir,snapnum,cosmo=cosmological, \
                        use_rundir=use_rundir,four_char=four_char,use_h0=h0);
                    have_h_stars=1;
                    hstars_should_concat=1;
                if (have==1):
                    m_all=np.concatenate((m_all,m)); c_all=np.concatenate((c_all,cc)); zm_all=np.concatenate((zm_all,zm));
                    x_all=np.concatenate((x_all,x)); y_all=np.concatenate((y_all,y)); z_all=np.concatenate((z_all,z)); 
                    vx_all=np.concatenate((vx_all,vx)); vy_all=np.concatenate((vy_all,vy)); vz_all=np.concatenate((vz_all,vz)); 
                    id_all=np.concatenate((id_all,id));
                    if(hstars_should_concat==1): h_all=np.concatenate((h_all,h)); 
                else:
                    m_all=m; x_all=x; y_all=y; z_all=z; zm_all=zm; c_all=cc; h_all=h;
                    vx_all=vx; vy_all=vy; vz_all=vz; id_all=id
                    have=1;

    dum = np.zeros(1)
    if (have != 1): return dum,dum,dum,dum,dum,dum,dum,dum,dum,dum,dum;
    if (m_all.size <= 1): return dum,dum,dum,dum,dum,dum,dum,dum,dum,dum,dum;
    
    if (BOX_MAX != 0):
        x=x_all-CENTER_BOX[0]; y=y_all-CENTER_BOX[1]; z=z_all-CENTER_BOX[2];
        ## make sure everything is cast correctly
        x=np.array(x,dtype='f'); y=np.array(y,dtype='f'); z=np.array(z,dtype='f')
        h_all=np.array(h_all,dtype='f'); c_all=np.array(c_all,dtype='f'); 
        m_all=np.array(m_all,dtype='f'); zm_all=np.array(zm_all,dtype='f')
        ok=ok_scan(c_all,pos=1) & ok_scan(m_all,pos=1) & ok_scan(h_all,pos=1) & ok_scan(zm_all,pos=1) & \
            ok_scan(x,xmax=BOX_MAX) & ok_scan(y,xmax=BOX_MAX) & ok_scan(z,xmax=BOX_MAX) & \
            (h_all < 100.) & (m_all > 1.0e-12);
        for vec in [id_all, m_all, x_all, y_all, z_all, vx_all, vy_all, vz_all, h_all, c_all, zm_all]:
            vec = vec[ok];

    h_all *= 1.25
    return id_all, m_all, x_all, y_all, z_all, vx_all, vy_all, vz_all, h_all, c_all, zm_all;


def build_snapshot_list( snapdir , use_h0=1,four_char=0,cosmological=0,skip_bh=0):
    i=0; imax=70; snums=[-1];
    while (i < imax):
        PPP_head = gadget.readsnap( snapdir,i,1,header_only=1,\
            h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh )
        if(PPP_head['k']==0):
            snums=np.concatenate((snums,np.array([i])));
            if (i > imax-50): imax += 50;
        i += 1;
    snums=snums[snums>=0];
    return snums


def get_camera_centering_filename( snapdir ):
    ## (first parse the directory names (for file-naming conventions))
    s0=snapdir.split("/"); snapdir_specific=s0[len(s0)-1]; n_s0=1;
    if(len(snapdir_specific)<=1): snapdir_specific=s0[len(s0)-2]; n_s0=2;
    ext = '.mv_cen'
    fname = snapdir+'/'+snapdir_specific+ext
    return fname


def build_camera_centering( snums, snapdir , 
    center_on_bh = 0, # center on first BH particle (good for nuclear sims)
    center_on_com = 0, # center on center of mass (good for merger sims)
    set_fixed_center = 0, # option to set a fixed center
    force_rebuild = 0, # force the routine to build from scratch (don't check for file)
    use_h0=1,four_char=0,cosmological=0,skip_bh=0):

    ## first check if the tabulated centers file already exists:
    fname = get_camera_centering_filename( snapdir )
    print(fname)
    if(force_rebuild==0): 
        if(os.path.exists(fname)): return 1
    #print('in build_camera_centering')
    #print(f'com? {center_on_com}')
    ## ok, if we're here, center-table doesn't exist, so let's build it:
    n_snums = np.array(snums).size - 1
    time_all = np.zeros( (n_snums) )
    cen = np.zeros( (n_snums,3) )
    outfi_cen = open(fname,'a')
    set_fixed_center = np.array(set_fixed_center)
    center_on_zero = 0
    if (set_fixed_center.size==1): 
        if (set_fixed_center==1): center_on_zero=1
    for i in range(n_snums):
        print('centering for snap ',snums[i],' in ',snapdir)
        PPP_head = gadget.readsnap( snapdir, snums[i], 0, header_only=1, \
            h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh )
        time_all[i] = PPP_head['time']
        cen[i,:] = [0.,0.,0.]
        

        if (center_on_zero==1): 
            continue
        elif (np.shape(set_fixed_center)[1]==3):
            cen[i,:]=set_fixed_center[i]
            continue

        elif (center_on_bh==1):
            PPP = gadget.readsnap( snapdir, snums[i], 5,\
                h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh )
            if (PPP['k']==1):
                if (PPP['m'].size > 0):
                    cen[i,:] = PPP['p'][0,:]
            continue
        
        elif (center_on_com==1):
            print('finding com')
            ptype = 1
            PPP = gadget.readsnap( snapdir, snums[i], ptype,\
                h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh )
            if (PPP['k']==1):
                if (PPP['m'].size > 0):
                    wt = np.array(PPP['m']); wt /= np.sum(wt); p = np.array(PPP['p'])
                    for j in [0,1,2]: cen[i,j]=np.sum(p[:,0]*wt)
            continue

        ## if not using any of the methods above, use our fancier iterative 
        ##   centering solution for the 'dominant' structure
        else:
            cen[i,:] = gadget.calculate_zoom_center( snapdir, snums[i], \
            cen=[0.,0.,0.], clip_size=2.e10, rho_cut=1.0e-5, \
            h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh)
            print('..zoom_cen at ',cen[i,:])
            sys.stdout.flush()
    
    ## all done, write out these results
    for i in range(len(cen[:,0])):
        v = np.array([ time_all[i], cen[i,0], cen[i,1], cen[i,2] ]).reshape(-1,4)
        np.savetxt(outfi_cen,v)
    outfi_cen.close()
    return 1
    

def get_precalc_zoom_center( snapdir, time_desired, cosmological=0 ):
    fname = get_camera_centering_filename( snapdir )
    cenfile = open(fname,'r'); brick=np.loadtxt(cenfile).reshape(-1,4)
    time=brick[:,0]; x0=brick[:,1]; y0=brick[:,2]; z0=brick[:,3]; ## define columns
    if (cosmological==1): 
        for q in [x0,y0,z0]: q /= time ## correct to comoving for spline

    nsmoo = 13
    windowfun = 'flat' ## boxcar smoothing
    #nsmoo = 21
    #windowfun = 'hanning' ## gaussian-like smoothing (should use larger window function)
    cen = np.zeros(3)
    for j,p in zip([0,1,2],[x0,y0,z0]):
        pp = util.smooth(p, window_len=nsmoo, window=windowfun)
        #pp = p ## linear interpolation (in case we don't have enough points) #DEBUG!
        if (cosmological==1): 
            pp_interp = time_desired * np.interp( np.log(time_desired), np.log(time), pp )
        else:
            pp_interp = np.interp( time_desired, time, pp )
        cen[j] = pp_interp

    print('... ... ... getting center for frame at t=',time_desired,' : ',cen)
    return cen
    
    

# builds the list of times to dump frames
def build_time_frame_grid( snapdir, snapshot_list, frames_per_gyr, \
        time_min=0, time_max=0, use_h0=1, four_char=0, cosmological=0, skip_bh=1 ):

    ## set times for frames
    if(time_min==0):
        PPP_head = gadget.readsnap( snapdir, snapshot_list[0], 1, header_only=1, \
            h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh )
        time_min = PPP_head['time'] ## set to first time in snapshot list
    if(time_max==0):
        PPP_head = gadget.readsnap( snapdir, snapshot_list[-1], 1, header_only=1, \
            h0=use_h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh )
        time_max = PPP_head['time'] ## set to last time in snapshot list
    if(cosmological==1):
        time_min = cosmological_time(time_min) ## input in 'a_scale', not time (Gyr)
        time_max = cosmological_time(time_max)

    time_frame_grid = np.arange(time_min, time_max, 1./np.float64(frames_per_gyr))

    if (cosmological==0): 
        a_scale_grid = time_frame_grid ## to avoid error below
    else:
        a_tmp = 10.**np.arange(-3.,0.001,0.001)
        t_tmp = cosmological_time(a_tmp)
        a_scale_grid = np.exp(np.interp(np.log(time_frame_grid),np.log(t_tmp),np.log(a_tmp)))
        a_scale_grid[a_scale_grid < a_tmp[0]] = a_tmp[0]

    return time_frame_grid, a_scale_grid


if __name__ == '__main__':
    movie_maker(xmax_of_box=size,snapdir=snapshot_dir,outputdir_master=output_dir,
        show_gasstarxray=type_to_make,cosmological=cosmo,
        frames_per_gyr=200.,four_char=0,
        i_snap_min=i_snap_min,i_snap_max=i_snap_max,skip_bh=1,center_on_com=0,set_fixed_center=set_fixed_center,
        theta_0=90.,phi_0=270.,add_extension='_p270_interp',
        camera_opening_angle=74.,
        pixels=500,z_to_add=0.02,
        use_old_extinction_routine=0)
