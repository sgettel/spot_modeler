#
# This code is based on SOAP by Boisse et al. (2012) and the spot modeling
# code described in Huerta et al. 2008. A grid is generated in the y-z plane
# (plane of the sky) where x is along the line of sight, y is rightward and z
# is up. The star is assumed to rotate as a solid body
#
# Optional Input:
#
# ngrid          : y-z grid is made of ngrid by ngrid points
#
# Input to implement:
# temp           : temperature of the star, used to select spectra - TBI
#
# Optional Input to implement:
# dtemp          : difference between stellar temp and spot temp 
# nr             : number of points used around the edge of the spot
# inc            : stellar inclination in degrees (90 = edge on, 0 = face on)
# nspot          : number of spots - currently just 1
# limb           : linear limb darkening coefficient
# spsize         : projected spot size in fraction of stellar radius
# vmax           : magnitude of rotational velocity in km/s - this does not include sin(inc)

# Created Sept. 2013 - SJG

import glob
import matplotlib.pyplot as plt
import numpy as np
import shelve
import sys
import vactoair as vc
from numpy.linalg import pinv
from scipy.signal import gaussian
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from gaussian_sjg import gaussian_sjg
from astropy.io import fits

# Setup the rest of the modules
def spot_modeler_main(temp=5400,dtemp=200, ngrid=300, nr=20, inc=90, nspot=1, limb=0.7, 
                        spsize=0.1, splat=0, splong=0, phase_step=15, phase_start=-180, 
                        vmax=2, nrot=1, saveplot='no', plotphase = 'no', 
                        lmin=6190,lmax=6290,savedat='no',savefits='no'):

    c = 2.99792458e8 #m/s

    #step between grid elements in units of stellar radius
    dgrid = 2./(ngrid - 1) 

    #make blank grid for star
    star, gpos, dstarr = star_grid(ngrid)
    
    #locate line of sight loop
    theta, xpos0, ypos0, zpos0 = spot_loop(nr, spsize)

    #put loop in initial spot position
    rm = make_position_matrix(inc, splat, splong,verbose='no')
    rminv = pinv(rm)
    xpos1, ypos1, zpos1 = spot_position(xpos0, ypos0, zpos0, rm)


    #calculate limb darkening once for whole star
    dark = limb_darkening(star, gpos, dstarr, limb)
    
    #calculate rotational broadening shifts once for whole star    
    shift = calc_rotation(star, gpos, vmax, inc)
    
    #load full star and spot spectra
    #stpath, sppath = find_spec_test()
    stpath, sppath = find_spec(temp=temp,dtemp=dtemp)
    wlst0, flst0, blst0, wlsp0, flsp0, blsp0 = read_spec(stpath, sppath)
    print len(wlst0),wlst0.min(),wlst0.max(), " from read_spec"
    
    #returns vacuum wavelengths
    vwl, flst, flsp, blst, blsp = select_range(wlst0, flst0, blst0, wlsp0, flsp0, blsp0,lmin,lmax)
    print len(vwl), vwl.min(),vwl.max(), " from select_range, with padding"    
    
    #convert to air wavelengths
    wl = vc.vactoairfn(vwl) #wavelengths of the stellar models
    print "Spectra loaded", len(wl),wl.min(),wl.max(), " air wavelengths"
    
    #linearly iterpolate on to 0.05A grid
    #fst = interp1d(wl,flst,kind='cubic')
    #fsp = interp1d(wl,flsp,kind='cubic')    
    
    #wl = np.linspace(lmin,lmax,num=(lmax-lmin)*20.)
    #flst = fst(wl)
    #flsp = fsp(wl)
    #wlstep = 0.05 #A
    wlstep = 0.0499866324374 #0.05A vac converted to air, mean over 4500-6800
    
    #phases = [0]

    nphases = np.fix(nrot*360/phase_step)
    #nphases = len(phases)
    photarr = np.zeros(nphases)
    phasearr = np.zeros(nphases)
    blank = np.zeros((ngrid, ngrid))
    totspecarr = np.zeros((len(wl),nphases))
    corrarr = np.zeros((len(wl),nphases)) 
    cntrarr = np.zeros(nphases)
    #bisarr = np.zeros((17,nphases))
    #ldrarr = np.zeros((12,nphases))
    bigarr = np.zeros((ngrid, ngrid, nphases))
    wholearr = np.zeros((ngrid, ngrid, nphases))
    print "Arrays setup"

    #Calculate stuff for blank star
    pltfile_bl = plot_file(180, spsize, splat, splong, inc, limb, nspot, ngrid, nr, blank='yes')
    whole_grid_bl, phot_bl = integrate_lc(dark, blank) 
    if plotphase == 'yes':
        plot_grid(dark, xpos0, ypos0, zpos0, blank, pltfile_bl, saveplot, blank='yes')
    specgrid_bl, lam0_bl = spec_grid(star, blank, wl, flst, flsp)
    print len(lam0_bl),lam0_bl.min(),lam0_bl.max()
    br_grid_bl = rot_broaden(specgrid_bl, lam0_bl, shift, star)
    totspec_bl = integrate_grid(br_grid_bl,dark)
    print "Blank star calculated"
    #Use blank star as template for cross-correlation
    template = totspec_bl/totspec_bl.max()
    
    #autocorrelation for the sake of completeness
    cor_bl = np.correlate(template, template, mode='same')
    print "Autocorrelation complete"
    
    #line_bisector needs updating
    #bis_bl, dlist = line_bisector(lam0_bl,totspec_bl)
    
    #ldr_bl = ldr_simple(lam0_bl,totspec_bl)
    
    
    for ph in range(nphases):
        
        #phase = phases[ph]
        phase = phase_start + ph*phase_step
        print 'phase = ', phase
        
        #xpos, ypos, zpos = rotate_simple(xpos0, ypos0, zpos0, phase, verbose='yes')
        xpos, ypos, zpos = rotate_star(xpos1, ypos1, zpos1, phase, inc)
        
        #Returns positions with x >= 0 or original positions if spot is hidden
        test, xpos_new, ypos_new, zpos_new = is_visible(xpos, ypos, zpos)
                
        if test: #if spot is present
        
            yflag, zflag = across_axis(nr, xpos_new, ypos_new, zpos_new)
            
            ygsmall, zgsmall, ygi, zgi =  small_grid(ypos_new, zpos_new, gpos, dgrid, yflag, zflag)

            spgrid = in_spot(ygsmall, zgsmall, phase, spsize, inc, rminv)
            print np.count_nonzero(spgrid), ' pixels in spot'
            
            
            big = small2big_grid(spgrid, ngrid, blank, ygi, zgi, gpos)
            #spot = 1, not-spot = 0
            
            pltfile = plot_file(phase, spsize, splat, splong, inc, limb, nspot, ngrid, nr)

            if plotphase == 'yes':
                plot_grid(dark, xpos_new, ypos_new, zpos_new, big, pltfile, saveplot)

            #combine limb darkening grid and big grid with spots
            whole_grid, phot = integrate_lc(dark, big)

            #store light curve
            photarr[ph] = phot
            phasearr[ph] = phase
            bigarr[:,:,ph] = big
            wholearr[:,:,ph] = whole_grid

    
            #feed unshifted spectra into grid
            specgrid, lam0 = spec_grid(star, big,wl,flst,flsp)
            #print len(lam0),lam0.min(),lam0.max()
            
            #shift spectra and linearly interpolate to same wavelength scale
            br_grid = rot_broaden(specgrid, lam0, shift, star)
            
            #integrate grid of spectra
            totspec = integrate_grid(br_grid,dark)
            totspecarr[:,ph] = totspec
            
            cor = np.correlate(template, totspec/totspec.max(), mode='same')
            corrarr[:,ph] = cor
            
            cntrarr[ph] = ccf_centroid(lam0, cor)
            #calculate bisectors
            #bis, dlist = line_bisector(lam0,totspec)
            #bisarr[:,ph] = bis
            
            #ldr = ldr_simple(lam0,totspec)
            #ldrarr[:,ph] = ldr
            
        else:
             
            photarr[ph] = phot_bl
            phasearr[ph] = phase
            totspecarr[:,ph] = totspec_bl    
            corrarr[:,ph] = cor_bl
            
            cntrarr[ph] = ccf_centroid(lam0_bl,cor_bl)
            #bisarr[:,ph] = bis_bl

            #ldrarr[:,ph] = ldr_bl
            bigarr[:,:,ph] = blank
        
            
    dl = cntrarr-np.median(cntrarr)
    rv = dl*c/lam0.mean()
    
    
    
    if savedat == 'yes':
        store_results(phasearr,photarr,lam0,totspecarr,corrarr,cntrarr,rv,wl, 
                        flst, flsp,temp,dtemp,ngrid, nr, inc, nspot, limb, spsize, splat, splong, phase_step,
                        phase_start, vmax, nrot, lmin, lmax)
                        
    if savefits == 'yes':
        write_fits(wlstep,phasearr,totspecarr,rv,wl,temp,dtemp,ngrid,nr,inc,nspot,limb,spsize,splat,splong,vmax,lmin,lmax)

    
    return lam0,totspecarr,phasearr,cntrarr,rv,photarr
    
#Test whether I'm using the laptop or desktop
def which_machine():
    comp = sys.platform
    if comp == 'darwin':
        print "Laptop detected"
        return 1
    elif comp == 'linux2':
        print "Desktop detected"
        return 2
    else:
        print "Unknown machine"
        return 0        
        
# Smarter finder program, will take stellar grid params
def find_spec(temp=5400,dtemp=200,grav=5.0,met=0.0): 


    temphi=temp/100.
    templo=(temp-dtemp)/100.
    
    if temphi >= 100:
        sthi = str(int(temphi))
    else:
        sthi = '0'+str(int(temphi))
        
    if templo >= 100:
         stlo = str(int(templo))
    else:
         stlo = '0'+str(int(templo))   
    
    
    g = '%s' % float('%.1g' % grav)
    m = '%s' % float('%.1g' % met)
    grav = float(g)
    met = float(m)
    
    print temphi, templo, sthi, stlo, grav, met
    
    comp = which_machine()
    specpath = set_inspec_path(comp)
    
    stpath = specpath+'lte'+str(sthi)+'-'+str(grav)+'-'+str(met)+'a+0.0.BT-Settl.7'
    sppath = specpath + 'lte' + str(stlo) + '-'+str(grav)+'-'+str(met)+'a+0.0.BT-Settl.7'
    
    #stpath = specpath+'lte058-5.0-0.0a+0.0.BT-Settl.7'
    #sppath = specpath+'lte056-5.0-0.0a+0.0.BT-Settl.7'

    return stpath, sppath
    
#set path to Phoenix spectra for given machine
def set_inspec_path(comp=1):
    
    if comp == 1:
        specpath = '/Users/Sara/cfasgettel-bulk/btsettl_ag09/'
    elif comp == 2:
        specpath = '/home/sgettel/cfasgettel-bulk/'
    else:
        print 'Unknown path'
        
    return specpath
    
#set path to output for a given machine
def set_output_path(comp=1):
    
    if comp == 1:
        outpath = '/Users/Sara/cfasgettel-bulk/spot_models/'
    elif comp == 2:
        outpath = '/home/sgettel/cfasgettel-bulk/spot_models/'
    else:
        print 'Unknown path'
    
    return outpath
        
#Stupid finder I can swap out easily     
def find_spec_test(): 
    comp = which_machine()
    specpath = set_inspec_path(comp)
    
    #stpath = specpath+'lte039-5.0-0.0a+0.0.BT-Settl.7'
    #sppath = specpath+'lte037-5.0-0.0a+0.0.BT-Settl.7'
    
    #stpath = specpath+'lte058-5.0-0.0a+0.0.BT-Settl.7'
    #sppath = specpath+'lte056-5.0-0.0a+0.0.BT-Settl.7'
    
    stpath = specpath+'lte054-5.0-0.0a+0.0.BT-Settl.7'
    #sppath = specpath+'lte052-5.0-0.0a+0.0.BT-Settl.7'
    stpath = specpath + 'lte049-5.0-0.0a+0.0.BT-Settl.7'
    
    
    #stpath = specpath+'lte045-5.0-0.0a+0.0.BT-Settl.7'
    #sppath = specpath+'lte040-5.0-0.0a+0.0.BT-Settl.7'
    
    return stpath, sppath
    
# Read the spectra for the blank star and the spot
def read_spec(stpath, sppath):
    #reads Phoenix spectra between 3600 and 9000A
    
    stfile = open(stpath)
    spfile = open(sppath)
        
    #split into string columns
    st0 = np.loadtxt(stfile,usecols=(0,1,2),dtype='str')
    sp0 = np.loadtxt(spfile,usecols=(0,1,2),dtype='str')
    st = np.zeros((st0.shape))
    sp = np.zeros((sp0.shape))
    
    #split string columns at exponent
    for i in range(st0.shape[0]):
        for j in range(st0.shape[1]):
            st[i,j] = dtoe(st0[i,j])
            
    for i in range(sp0.shape[0]):
        for j in range(sp0.shape[1]):
            sp[i,j] = dtoe(sp0[i,j])
    
    wlst = st[:,0]
    flst = 10**(st[:,1] - 8.)
    blst = 10**(st[:,2] - 8.)  
      
    a = np.argsort(wlst)
    wlst = wlst[a]
    flst = flst[a]
    blst = blst[a]
    
    wlsp = sp[:,0]
    flsp = 10**(sp[:,1] - 8.)
    blsp = 10**(sp[:,2] - 8.)
    
    b = np.argsort(wlsp)     
    wlsp = wlsp[b]
    flsp = flsp[b]
    blsp = blsp[b]
    
    #keep only the likely regions of the spectrum
    c = np.squeeze(np.where((wlst >= 3600.0) & (wlst <= 9000.0)))
    d = np.squeeze(np.where((wlsp >= 3600.0) & (wlsp <= 9000.0)))
    
    wlst = wlst[c]
    flst = flst[c]
    blst = blst[c]
    
    wlsp = wlsp[d]
    flsp = flsp[d]
    blsp = blsp[d]
    
    return wlst, flst, blst, wlsp, flsp, blsp

        
# Function to convert strings in the format 1.234D+05 into floats    
def dtoe(instr):   
     pieces = instr.split('D')
     
     flt = '%E' % float(pieces[0]+'E'+pieces[1])
     
     return flt
     
#cut down spectra        
def select_range(wlsta, flsta, blsta, wlspa, flspa, blspa, lmin=6190,lmax=6290):
    
    in1 = np.squeeze(np.where((wlsta >= lmin) & (wlsta < lmax)))
    in2 = np.squeeze(np.where((wlspa >= lmin) & (wlspa < lmax)))
    print len(in1), len(in2), ' total phoenix wl in range'
    
    wlst = wlsta[in1]
    flst = flsta[in1]
    blst = blsta[in1]
    wlsp = wlspa[in2]
    flsp = flspa[in2]
    blsp = blspa[in2]
    
    #Want spacing 0.05A, prune excess values
    diffst = wlst*100 % 5
    diffsp = wlsp*100 % 5
    kt1 = np.squeeze(np.where(diffst < 1.0e-9))
    kt2 = np.squeeze(np.where(abs(diffst-5.0) < 1.0e-9))
    kp1 = np.squeeze(np.where(diffsp < 1.0e-9))
    kp2 = np.squeeze(np.where(abs(diffsp-5.0) < 1.0e-9))
    
    kt = np.append(kt1,kt2)
    kp = np.append(kp1,kp2)
    kt = np.sort(kt) 
    kp = np.sort(kp) 
    
    #kt = np.squeeze(np.where(wlst*100 % 5 == 0))
    #kp = np.squeeze(np.where(wlsp*100 % 5 == 0))
    print len(kt),len(kp), ' in step of 0.05'    
    
    wlst = wlst[kt]
    wlsp = wlsp[kp]
    flst = flst[kt]
    flsp = flsp[kp]
    blst = blst[kt]
    blsp = blsp[kp]
        
    #Cut duplicate values from stellar arrays
    d1 = np.linspace(0, len(kt)-2, num=len(kt)-1)
    d2 = np.linspace(1, len(kt)-1, num=len(kt)-1)
    d1 = d1.astype(int)
    d2 = d2.astype(int)
    x = np.squeeze(np.where(wlst[d2] == wlst[d1]))
    wl = np.delete(wlst,d2[x])
    flst = np.delete(flst,d2[x])
    blst = np.delete(blst,d2[x])
    
    #repeat for spot arrays
    d1 = np.linspace(0, len(kp)-2, num=len(kp)-1)
    d2 = np.linspace(1, len(kp)-1, num=len(kp)-1)
    d1 = d1.astype(int)
    d2 = d2.astype(int)
    x = np.squeeze(np.where(wlsp[d2] == wlsp[d1]))
    wlsp = np.delete(wlsp,d2[x])
    flsp = np.delete(flsp,d2[x])
    blsp = np.delete(blsp,d2[2])
    
    
    return wl, flst, flsp, blst, blsp
    
# Apply PSF, turbulence to spectra to match observations
#def real_spec():




# Make a single gaussian line to use as a test spectrum - obselete
def line_model(cont=40000.0, ld=0.5, lambda0=6200.0, sig=0.05, npix=200, dlambda=0.005):
    #dlambda = 0.005 A
    psig = sig/dlambda
    spec = 1. - gaussian(npix,psig)*ld #in pixels
    pix = np.linspace(0,npix-1, num=npix)
    lam = pix*dlambda+lambda0
    
    return lam, spec
    
def line_comb(nlines=3, lambda0=6200.0, sig=0.05, npix=200, dlambda=0.005,ld=0.5):
    #A simplistic routine to append single line models together - never actually used this

    lamgrid = []
    spgrid = []

    for i in range(nlines):
        lami, speci = line_model(lambda0=lambda0,sig=sig,npix=npix, dlambda=dlambda,ld=ld)
        lamgrid = np.concatenate((lamgrid,lami+dlambda*npix*i))
        spgrid = np.concatenate((spgrid,speci))    
        
    return lamgrid, spgrid
    
# Make blank star with uniform intensity 1.0
def star_grid(ngrid):

    #make array of positions in normalized y-z space
    star = np.zeros((ngrid, ngrid))
    dstarr = np.zeros((ngrid, ngrid))
    gpos = np.linspace(-1.0, 1.0, num=ngrid)
    for y in range(ngrid):
        for z in range(ngrid):
            #dst is also sin(theta) where theta is angle from line of sight
            dst = np.sqrt(gpos[y]**2 + gpos[z]**2) #distance from center of grid
            dstarr[z, y] = dst
            if dst <= 1.0:
                star[z, y] = 1.0
    return star, gpos, dstarr

# Generate default spot of a given size
# These are always centered on the x-axis to start
def spot_loop(nr, sprad):

    step_angle = 2*np.pi/nr #in radians
    init_angle = 0. 

    #make position arrays
    theta = np.zeros(nr)
    xpos = np.ones(nr)*np.sqrt(1.0 - sprad**2) #constant in this position only
    ypos = np.zeros(nr)
    zpos = np.zeros(nr)

    for step in range(nr):
        theta[step] = init_angle + step*step_angle
        ypos[step] = sprad*np.cos(theta[step])
        zpos[step] = sprad*np.sin(theta[step])
#        print theta[step], xpos[step], ypos[step], zpos[step]
    return theta, xpos, ypos, zpos

#Move spot loop to the correct lat-long
def spot_position(xpos, ypos, zpos, rm):
    
    xpos_new = np.zeros(xpos.size)
    ypos_new = np.zeros(xpos.size)
    zpos_new = np.zeros(xpos.size)
    
    for el in range(xpos.size):
        
        pos = [xpos[el], ypos[el], zpos[el]]
        post = np.transpose(pos)
        pos_new = np.dot(rm, post)
        
        xpos_new[el] = pos_new[0]
        ypos_new[el] = pos_new[1]
        zpos_new[el] = pos_new[2]
    
    return xpos_new, ypos_new, zpos_new

#from SOAP paper, eqn. 1, with corrections...
def make_position_matrix(inc, splat, splong,verbose='no'):
    dphi = 90. - inc #this runs 0-180 deg
    phi = dphi*np.pi/180.    #degrees to radians
    dpsi = -1.*splat
    psi = dpsi*np.pi/180.
    theta = splong*np.pi/180.
    
    #This is a YZY rotation matrix
    rm = np.zeros((3, 3))
    #top row
    rm[0,0] = np.cos(phi)*np.cos(theta)*np.cos(psi) - np.sin(phi)*np.sin(psi)
    rm[0,1] = -np.cos(phi)*np.sin(theta)
    rm[0,2] = np.cos(psi)*np.sin(phi) + np.cos(phi)*np.cos(theta)*np.sin(psi)
    #middle row
    rm[1,0] = np.cos(psi)*np.sin(theta)
    rm[1,1] = np.cos(theta)
    rm[1,2] = np.sin(theta)*np.sin(psi)
    #bottom row
    rm[2,0] = -np.cos(phi)*np.sin(psi) - np.cos(theta)*np.cos(psi)*np.sin(phi)
    rm[2,1] = np.sin(phi)*np.sin(theta)
    rm[2,2] = np.cos(phi)*np.cos(psi) - np.cos(theta)*np.sin(phi)*np.sin(psi)
    
    if verbose == 'yes':
        print rm
    
    return rm

#Figure out if any part of the spot is visible
def is_visible(xpos, ypos, zpos):
    #print xpos
    show = np.squeeze(np.where(xpos > 0))
    
    if show.size == 0 :
        print "Spot not visible"
        return False, xpos, ypos, zpos
    else:
        print "Spot is visible"
        return True, xpos[show], ypos[show], zpos[show]

#Figure out if the spot falls across either the y or z-axis
def across_axis(nr, xpos, ypos, zpos):
    
    show = np.squeeze(np.where(xpos > 0))
    if show.size > 0:

    
        #Fewer points means part of the spot is not visible
        if xpos.size < nr:
            print "Spot partially visible"
        
            yneg = np.squeeze(np.where(ypos < 0))
            zneg = np.squeeze(np.where(zpos < 0))

            #Test if there are both positive and negative points
            if (not yneg.size == 0) and (not yneg.size  == len(ypos)):
                zflag = 1
                print "Spot crosses z-axis"
            else:
                zflag = 0
            
            if (not zneg.size == 0) and (not zneg.size == len(zpos)):
                yflag = 1
                print "Spot crosses y-axis"
            else:
                yflag = 0
                
        else:
            print "Full spot is visible"
            yflag = 0
            zflag = 0

    else:
        yflag = 0
        zflag = 0

    return yflag, zflag

#Make a small grid in the y-z plane to contain the spot
#Only points on front of star are passed in
def small_grid(ypos, zpos, gpos, dgrid, yflag, zflag):

    if yflag == 0:
        #normal range is okay
        yrange = [min(ypos),max(ypos)]
    else:
        #if abs(min(ypos)) > abs(max(ypos)): #previous test, fails when they are equal
        if max(ypos) < 0:
            #spot crosses negative y-axis
            yrange = [-1, max(ypos)]
        else:
            #spot crosses positive y-axis
            yrange = [min(ypos), 1]
        
            
    if zflag == 0:
        #normal range is okay
        zrange = [min(zpos),max(zpos)]
    else:
        #if abs(min(zpos)) > abs(max(zpos)): #previous test, fails when they are equal
        if max(zpos) < 0:
            #spot crosses negative y-axis
            zrange = [-1, max(zpos)]
        else:
            #spot crosses positive y-axis
            zrange = [min(zpos), 1]
    
    #print min(ypos),max(ypos)
    #print yrange
    #print zrange
     
    #find the grid positions that fall inside the rectangle bounding the spot
    smy = np.squeeze(np.where((gpos <= yrange[1]+dgrid) & (gpos >= yrange[0]-dgrid)))
    smz = np.squeeze(np.where((gpos <= zrange[1]+dgrid) & (gpos >= zrange[0]-dgrid)))
    
    smypos = gpos[smy]
    smzpos = gpos[smz]

    return smypos, smzpos, smy, smz
    
# Figure out which cells are part of the spots
#Only points on front of star are passed in
def in_spot(ygpos, zgpos, phase, spsize, inc, rminv):
    #spot cells = 1, non-spot = 0
    
    xtestl = []
    ytestl = []
    ztestl = []

    spgrid = np.zeros((len(zgpos), len(ygpos)))
    print spgrid.size, ' = small grid size'
    for y in range(len(ygpos)):
        for z in range(len(zgpos)):

            #check if cell is on star
            dsq = ygpos[y]**2 + zgpos[z]**2 
            
            if dsq <= 1:
                
                xg = np.ones(1)*np.sqrt(1. - dsq)
                yg = np.ones(1)*ygpos[y]
                zg = np.ones(1)*zgpos[z]
                
                #undo phase rotation
                xg2, yg2, zg2 = rotate_star(xg, yg, zg, -1*phase, inc)
                
                #undo spot positioning...
                pos2 = [xg2, yg2, zg2]
                
                pos_center = np.dot(rminv,pos2)
                xtest = pos_center[0]
                ytest = pos_center[1]
                ztest = pos_center[2]
                
                #print xg, xtest
                xtestl.extend(xtest)
                ytestl.extend(ytest)
                ztestl.extend(ztest)
                
                #print xtest, ytest, ztest, ytest**2 + ztest**2
                #Test if rotated cell is inside original spot
                if ytest**2 + ztest**2 <= spsize**2:
                    spgrid[z, y] = 1

    return spgrid

# Impose grid of spot on grid of whole star
def small2big_grid(spgrid, ngrid, big0, ygi, zgi, gpos):

    big = np.zeros((ngrid, ngrid)) #spot cell = 1

    for y in range(len(ygi)):
        for z in range(len(zgi)):
            big[zgi[z],ygi[y]] = spgrid [z,y]
    
    #Add to previous grid
    big += big0  #This will break if there are overlapping spots!

    return big


# Load spot/non-spot spectra into grid as appropriate
# This considers temperature but not rotational shifts
def spec_grid(star, big, lam, star_spec, spot_spec):
    
    
    #make grid of ngrid x ngrid x nwavelengths
    spec_grid = np.zeros((star.shape[0], star.shape[0], len(lam)))
    
    #Is there a better way to do this?
    for y in range(star.shape[0]):
        for z in range(star.shape[0]):
            if star[z,y] == 1:
                if big[z,y] >= 1: #this is a spot cell
                    spec_grid[z,y,0:] =  spot_spec
                else:            
                    spec_grid[z,y,0:] = star_spec
                    
    return spec_grid, lam

# Load spot/non-spot gaussians into grid as appropriate
# This considers temperature but not rotational shifts
def spec_grid_simple(star, big):
    
    lam, star_spec = line_model() #ld=0.5
    lam, spot_spec = line_model(ld=0.2) #shallower lines b/c less flux
    
    #make grid of ngrid x ngrid x nwavelengths
    spec_grid = np.zeros((star.shape[0], star.shape[0], len(lam)))

    #Is there a better way to do this?
    for y in range(star.shape[0]):
        for z in range(star.shape[0]):
            if star[z,y] == 1:
                if big[z,y] >= 1: #this is a spot cell
                    spec_grid[z,y,0:] =  spot_spec
                else:            
                    spec_grid[z,y,0:] = star_spec
                    
    return spec_grid, lam
        

# Calculate doppler shift for each cell
def calc_rotation(star, gpos, vmax, inc):
    print "Vmax = ", vmax
    
    inc_r = inc*np.pi/180. #radians
    
    rot_grid = np.zeros((len(gpos),len(gpos)))
    
    
    for y in range(len(gpos)):
        for z in range(len(gpos)):
            vx = vmax*gpos[y]*np.sin(inc_r)*star[z, y]
            rot_grid[z,y] = vx
    
    return rot_grid


#apply broadening to spectra in grid
def rot_broaden(specgrid, lam0, rot_grid, star):
    
    specgrid_new = np.zeros(specgrid.shape)
    
    #Pad wavelength array - make this a separate function and less klunky...
    lamminus = lam0.min()*np.ones(100) - np.linspace(100,1,num=100)/100.
    lamplus = lam0.max()*np.ones(100) + np.linspace(1,100,num=100)/100.
    lammore = np.concatenate((lamminus,lam0))#, lamplus) #extend initial array
    lammore = np.concatenate((lammore,lamplus))
    
    #rot_grid is projected velocity in km/s
    for y in range(specgrid.shape[1]):
        for z in range(specgrid.shape[0]):
            if star[z,y] == 1:
            
                #convert velocity to wavelengths and shift wavelength array
                lami = (rot_grid[z,y]/2.99792458e5 + 1)*lammore   
                
                #pad spectrum array
                specmore = np.concatenate((np.ones(100),specgrid[z,y,0:]))
                specmore = np.concatenate((specmore,np.ones(100)))
                
                #interpolate onto original wavelength scale       
                f = interp1d(lami, specmore)
                spec_new = f(lam0)
                specgrid_new[z,y, 0:] = spec_new
            
    return specgrid_new

# Calculate limb darkening weight for each cell
def limb_darkening(grid, gpos, dstarr, limb):
    # I(mu)/I_0 = 1 - c(1 - mu), where mu = cos(theta), c = limb coefficient
    #dstarr = sin(theta)
    theta = np.zeros((len(gpos), len(gpos)))
    mu = np.zeros((len(gpos), len(gpos)))
    limb_scale = np.zeros((len(gpos), len(gpos)))
    
    for y in range(len(gpos)):
        for z in range(len(gpos)):
            if grid[z, y] == 1:
                theta[z, y] = np.arcsin(dstarr[z, y])
                #mu[z, y] = np.cos(theta[z, y])
                mu[z, y] = np.sqrt(1 - dstarr[z, y]**2)
                limb_scale[z, y] = 1.0 - limb*(1.0 - mu[z, y])
    return limb_scale

# Plot the stellar grid itself, plus spot & spot grid
def plot_grid(grid, xpos0, ypos0, zpos0, big, filename, saveplot='no', blank='no'):
    plt.figure(1)
    plt.imshow(grid, cmap=plt.cm.hot, extent=[-1, 1, -1, 1],origin='lower')
    #sz = grid.shape[0] #find size of the grid

    if not blank == 'yes': #if star not known to be blank

        #Figure out how much of the spot is visible
        test, xpos, ypos, zpos = is_visible(xpos0, ypos0, zpos0)

    
        #Overplot grid of spots
        plt.imshow(big, cmap=plt.cm.gray_r, extent=[-1,1,-1,1], alpha=0.2,origin='lower')

        if test == True:
           #Overplot  spot loop
           plt.scatter(ypos,zpos)

    if saveplot == 'yes':
    #plt.savefig('/home/sgettel/research/spot_simulation/'+filename)
        plt.savefig('/Users/Sara/Dropbox/cfasgettel/research/spot_simulation/'+filename)
    
    plt.close(1)

# Generate filename for plot
def plot_file(phase, spsize, splat, splong, inc, limb, nspot, ngrid, nr, blank='no'):
    
    #file format phase_spsize_splat_splong_inc_limb_nspot_ngrid_nr
    if blank == 'yes':
        string = 'grid_plot_'+str(phase)+'_'+str(spsize)+'_'+str(splat)+'_'+str(splong)+'_'+str(inc)+'_'+str(limb)+'_'+str(nspot)+'_'+str(ngrid)+'_'+str(nr)+'_blank.png'
    else:
        string = 'grid_plot_'+str(phase)+'_'+str(spsize)+'_'+str(splat)+'_'+str(splong)+'_'+str(inc)+'_'+str(limb)+'_'+str(nspot)+'_'+str(ngrid)+'_'+str(nr)+'.png'
    return string


#Advance the phase of the stellar rotation
def rotate_star(xpos1, ypos1, zpos1, drph, dinc, verbose='no'):
    #rotate star around the stellar inclination unit vector
    
    inc = dinc*np.pi/180.
    rph = drph*np.pi/180.
    
    uvec = [np.cos(inc), 0., np.sin(inc)]
    #print uvec
    
    alpha = 1. - np.cos(rph)
    beta = np.sin(rph)
    
    xpos_new = np.zeros(xpos1.size)
    ypos_new = np.zeros(xpos1.size)
    zpos_new = np.zeros(xpos1.size)
    
    rm = np.zeros([3,3])
    rm[0,0] = alpha*uvec[0]**2 + np.cos(rph)
    rm[0,1] = -beta*uvec[2]         #additional y term = 0
    rm[0,2] = alpha*uvec[0]*uvec[2] #same
    rm[1,0] = beta*uvec[2]          #same
    rm[1,1] = np.cos(rph)           #same
    rm[1,2] = -beta*uvec[0]         #same
    rm[2,0] = alpha*uvec[0]*uvec[2] #same
    rm[2,1] = beta*uvec[0]          #same
    rm[2,2] = alpha*uvec[2]**2 + np.cos(rph)
    
    if verbose == 'yes':
        print rm
    
    for el in range(xpos1.size): #all position arrays have the same size
        
        pos = [xpos1[el], ypos1[el], zpos1[el]]
        #pos_new = np.dot(pos, rm)
        post = np.transpose(pos)
        pos_new = np.dot(rm,post)
        
        xpos_new[el] = pos_new[0]
        ypos_new[el] = pos_new[1]
        zpos_new[el] = pos_new[2]
    
    return xpos_new, ypos_new, zpos_new

#Advance the phase of the star by rotating spot positions around z-axis - only valid if inc=90
def rotate_simple(xpos, ypos, zpos, dphase,verbose='no'):

    phase = dphase*np.pi/180. #degrees to radians

    if verbose == 'yes':
        print phase
    
    xpos_new = np.zeros(xpos.size)
    ypos_new = np.zeros(xpos.size)
    zpos_new = np.zeros(xpos.size)

    #make a z-axis rotation matrix
    rm = np.zeros((3, 3))
    rm[0,0] = np.cos(phase)
    rm[1,1] = np.cos(phase)
    rm[2,2] = 1.
    rm[0,1] = np.sin(phase)
    rm[1,0] = -1*np.sin(phase)

    if verbose == 'yes':
        print rm
    
    for el in range(xpos.size):
        
        pos = [xpos[el], ypos[el], zpos[el]]
        pos_new = np.dot(pos, rm)
        
        xpos_new[el] = pos_new[0]
        ypos_new[el] = pos_new[1]
        zpos_new[el] = pos_new[2]

    return xpos_new, ypos_new, zpos_new

#Sum photometry over all cells
def integrate_lc(dark_grid, spot_grid):

    #print dark_grid.shape
    #print spot_grid.shape
    star_grid = 1 - spot_grid #Now spot = 0, not-spot = 1
    
    whole_grid = dark_grid*star_grid
    phot = np.sum(whole_grid)
    return whole_grid, phot
    
#sum spectra over all cells
def integrate_grid(spgrid0,dark_grid):

    print spgrid0.shape
    spgrid = np.zeros(spgrid0.shape)
    

    ##first apply limb darkening
    #for i in range(spgrid0.shape[2]): #must be a smarter way to do this
    #    spgrid[:,:,i] = dark_grid*spgrid0[:,:,i]
    
    #integrate over grid
    totspec = np.sum(spgrid0, axis=0) #can use a tuple here instead
    totspec = np.sum(totspec, axis=0)    
    return totspec


#calculate centroid for CCF
def ccf_centroid(lam0, cor):
    
    lpeak = np.argmax(cor) #find highest point    
    fsamp = cor[lpeak-3:lpeak+4] #include the 3 points on either side
    lsamp = lam0[lpeak-3:lpeak+4]
    
    #fit a parabola
    fit = np.polyfit(lsamp,fsamp,2) 
    
    #find axis of symmetry h = -b/(2a) where y = ax**2 + bx + c
    cntr = -fit[1]/(2*fit[0])
    
    return cntr
    
#calculate bisector for CCF
#def line_bisector(lam0, cor):

#find lines automatically
#def line_finder():
    
#calculate line depth ratios for selected lines - obselete            
def ldr_simple(wl,sp,contsamp=5,fitsamp=0.25,dlambda=0.05,plot_lines='no'):
    #translation of fit_ldr.pro into python...
    
    #c = 2.99792458e8
    
    #read line list

    comp = which_machine()
    if comp == 1:
        lfile = open('/Users/Sara/Dropbox/cfasgettel/research/harpsn/linelists/biazzo07.dat')
    elif comp == 2:
        lfile = open('/home/sgettel/Dropbox/cfasgettel/research/harpsn/linelists/biazzo07.dat')
    
   
    ldat = np.loadtxt(lfile,usecols=(0,4,5)) 
    rline = ldat[:,0]
    num = ldat[:,1]
    #blend = ldat[:,2]
    
#    #put lines into pairs...skipping the blends for now
    p=np.array([[1,2],[5,4],[9,8],[11,10],[12,14],[13,15],[13,16],[17,18],[21,19],
        [23,22],[24,25],[26,25]])
    #,[3,6],[7,6],[21,20]
    ldr = np.zeros(p.shape[0])
    #eldr = np.zeros(p.shape[0])
    
    #select continuum region
    cfix=np.array([[6197.0,6197.4],[6201.8,6202.2],[6211.0,6211.4],[6217.0,6217.3],
        [6221.8,6222.2],[6225.6,6225.8],[6228.6,6228.8],[6234.1,6234.5],
        [6241.0,6241.5],[6244.8,6245.1],[6241.0,6241.5],[6248.3,6248.8],
        [6241.0,6241.5],[6248.3,6248.8],[6248.3,6248.8],[6255.4,6255.7],
        [6255.4,6255.7],[6260.1,6260.4],[6264.1,6264.3],[6266.5,6266.7],
        [6267.1,6267.4],[6270.5,6271.0],[6269.5,6269.9],[6275.0,6275.3]])

    for pairs in range(p.shape[0]):
        r1 = np.squeeze(np.where(num == p[pairs,0]))
        r2 = np.squeeze(np.where(num == p[pairs,1]))
        wl1 = rline[r1]
        wl2 = rline[r2]
        mid = (wl2 - wl1)/2. + wl1
        
        #locate continuum around line pair - for filtered continuum
        contrange = [mid - contsamp, mid + contsamp]
        chunk = np.squeeze(np.where((wl > contrange[0]) & (wl < contrange[1])))
       
        
        #calculate fixed continuum - this is just a line
        chunk1 = np.squeeze(np.where((wl > cfix[2*pairs,0]) & (wl < cfix[2*pairs,1])))
        chunk2 = np.squeeze(np.where((wl > cfix[2*pairs+1,0]) & (wl < cfix[2*pairs+1,1])))
        cont1 = np.median(sp[chunk1])
        cont2 = np.median(sp[chunk2])
        x = [np.median(wl[chunk1]),np.median(wl[chunk2])]
        y = [cont1, cont2]
        #equation of the form y = m*x + b
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A,y)[0]
        
        #find approximate line position
        findrange1 = [wl1 - fitsamp, wl1 + fitsamp]
        findrange2 = [wl2 - fitsamp, wl2 + fitsamp]
        line1=np.squeeze(np.where((wl > findrange1[0]) & (wl < findrange1[1])))
        line2=np.squeeze(np.where((wl > findrange2[0]) & (wl < findrange2[1])))
        
        
        #find measured line minima
        ss1 = np.argmin(sp[line1])
        ss2 = np.argmin(sp[line2])
        
        fitrange1 = [wl[line1[ss1]] - fitsamp, wl[line1[ss1]] + fitsamp]
        fitrange2 = [wl[line2[ss2]] - fitsamp, wl[line2[ss2]] + fitsamp]
        
        newline1=np.squeeze(np.where((wl > fitrange1[0]) & (wl < fitrange1[1])))
        newline2=np.squeeze(np.where((wl > fitrange2[0]) & (wl < fitrange2[1])))
        
        x1=wl[newline1]
        y1=sp[newline1]

        x2=wl[newline2]
        y2=sp[newline2]
        
        #fit Gaussian to line center with curve_fit
        popt1, pcov1 = curve_fit(gaussian_sjg,x1,y1)
        popt2, pcov2 = curve_fit(gaussian_sjg,x2,y2)
        #yfit1 = gaussian_sjg(x1,popt1[0],popt1[1],popt1[2],popt1[3])
        #yfit2 = gaussian_sjg(x2,popt2[0],popt2[1],popt2[2],popt2[3])

        
        #find fitted line minimum - nope, this is still measured line minimum
        nmin1 = np.min(sp[newline1])
        nss1 = np.argmin(sp[newline1])
        nmin2 = np.min(sp[newline2])
        nss2 = np.argmin(sp[newline2])
        
        wmin1 = wl[newline1[nss1]] #len = 1
        wmin2 = wl[newline2[nss2]]
        
        contfix1 = m*wmin1 + b #len = 1
        contfix2 = m*wmin2 + b
        
        #ccs1 = np.squeeze(np.where(wl[chunk] == wmin1))
        #ccs2 = np.squeeze(np.where(wl[chunk] == wmin2))
        
        #calculate line depth ratio using fixed continuum
        ldfix1 = (contfix1 - nmin1)/contfix1
        ldfix2 = (contfix2 - nmin2)/contfix2
        ldrfix = ldfix1/ldfix2
        ldr[pairs] = ldrfix
        
        #estimate errors
        #econt = np.sqrt(np.median([cont1,cont2]))
        
#        if plot_lines == 'yes':
#            plt.plot(wl[chunk],sp[chunk])
#            plt.xlim(contrange)
            
    return ldr
        
#calculate line bisector for single gaussian - obselete
def line_bisector_test(lam0,specarr):
    
    dlist = np.arange(10, 95, step = 5)/100.
    #print dlist
    
    #transform to unit line depth
    emspec = 1 - (specarr/specarr.max())
    stretch = emspec/emspec.max()
    normspec = 1 - stretch
        
    #divide line/ccf in half at minimum - this will break if the line shape is really weird 
    #Could take the derivative instead and count zeros
    #swap over x=y to interpolate
    mns = np.argmin(normspec)    
    sleft = normspec[0:mns]
    sright = normspec[mns:]
    lleft = lam0[0:mns]
    lright = lam0[mns:]
    
    #reverse left arrays
    sleft.sort()
    lleft = -lleft
    lleft.sort()
    lleft = -lleft
              
    fl = interp1d(sleft,lleft)
    fr = interp1d(sright,lright)
    
    lamnewl = fl(dlist) 
    lamnewr = fr(dlist)
    
    #print lamnewl    
    #print lamnewr
        
    bis = (lamnewr-lamnewl)/2 + lamnewl   
    #span = (lamnewr-lamnewl)
            
    return bis, dlist
            
#shelve data so I don't need to re-run the simulation            
def store_results(phasearr,photarr,lam0,totspecarr,corrarr,cntrarr,rv, wl, flst, flsp,temp=5400, dtemp=200,ngrid=300, 
                    nr=20, inc=90, nspot=1, limb=0.7, spsize=0.1, splat=0, splong=0,
                    phase_step=15, phase_start=0, vmax=2, nrot=1,lmin=6190,lmax=6290):       
            
     #file format temp/dtemp_spsize_lmin_lmax_vmax_splat_splong_inc_limb_phstep_phstart_nrot_nspot_ngrid_nr       
            
    comp = which_machine()   
    rpath0 = set_output_path(comp)  
    rpath = rpath0+'T'+str(temp)+'/'
    
    tag = str(dtemp) + '_' + str(spsize)+'_'+str(lmin)+'_'+str(lmax)+'_'+str(vmax)+'_'+str(splat)+'_'+str(splong)+'_'+str(inc)+'_'+str(limb)+'_'+str(phase_step)+'_'+str(phase_start)+'_'+str(nrot)+'_'+str(nspot)+'_'+str(ngrid)+'_'+str(nr)
                    
    print rpath+'shelve_model_'+tag                          
    store = shelve.open(rpath+'shelve_model_'+tag)
    store['phasearr'] = phasearr
    store['photarr'] = photarr
    store['specarr'] = totspecarr
    store['corrarr'] = corrarr
    store['cntrarr'] = cntrarr
    #store['ldrarr'] = ldrarr
    store['rv'] = rv
    store['wl'] = wl
    store['flst'] = flst
    store['flsp'] = flsp
    store.close()
    print "saving model data"   
            
#load previous simulation
def restore_results(filename,temp=5400):
    #file format (dtemp)_spsize_lmin_lmax_vmax_splat_splong_inc_limb_phstep_phstart_nrot_nspot_ngrid_nr       
            
    comp = which_machine()     
    rpath0 = set_output_path(comp)  
    rpath = rpath0+'T'+str(temp)+'/'
    print rpath+filename
                
    stored = shelve.open(rpath+filename)
    phasearr = stored['phasearr']
    photarr = stored['photarr'] 
    totspecarr = stored['specarr'] 
    corrarr = stored['corrarr'] 
    cntrarr = stored['cntrarr'] 
    #ldrarr = stored['ldrarr'] 
    rv = stored['rv']
    wl = stored['wl']
    flst = stored['flst']
    flsp = stored['flsp'] 
    stored.close()
    
    return phasearr, photarr, totspecarr, corrarr, cntrarr, rv, wl, flst, flsp


def write_fits(wlstep,phasearr,totspecarr,rv,wl,temp,dtemp,ngrid,nr,inc,nspot,limb,spsize,splat,splong,vmax,lmin,lmax):
    
    #file format temp/dtemp_spsize_lmin_lmax_vmax_splat_splong_inc_limb_phase_nspot_ngrid_nr
    comp = which_machine()   
    rpath0 = set_output_path(comp)  
    rpath = rpath0+'T'+str(temp)+'/fits/'
    
    
    for i,ph in enumerate(phasearr):

        tag = str(dtemp) + '_' + str(spsize)+'_'+str(lmin)+'_'+str(lmax)+'_'+str(vmax)+'_'+str(splat)+'_'+str(splong)+'_'+str(inc)+'_'+str(limb)+'_'+str(ph)+'_'+str(nspot)+'_'+str(ngrid)+'_'+str(nr)
  
        #from http://docs.astropy.org/en/stable/io/fits/index.html
        hdu = fits.PrimaryHDU(totspecarr[:,i])
        hdulist = fits.HDUList([hdu])
        prihdr = hdulist[0].header
        prihdr['CDELT1'] = wlstep
        prihdr['CRVAL1'] = wl.min()
        prihdr['PHASE'] = ph
        prihdr['RVCALC'] = rv[i]
        prihdr['TEMP'] = temp
        prihdr['NGRID'] = ngrid
        prihdr['NR'] = nr
        prihdr['INC'] = inc
        prihdr['NSPOT'] = nspot
        prihdr['LIMB'] = limb
        prihdr['SPSIZE'] = spsize
        prihdr['SPLAT'] = splat
        prihdr['SPLONG'] = splong
        prihdr['VMAX'] = vmax

        hdulist.writeto(rpath+'fits_model'+tag+'.fits',clobber=True)
        
def read_fits(fitsname,temp=5400):
    comp = which_machine()   
    rpath0 = set_output_path(comp)  
    rpath = rpath0+'T'+str(temp)+'/fits/'
        
    fname = rpath + fitsname
    hdulist = fits.open(fname)
    
    #header = hdulist[0].header
    #keys0 = header.keys()        
        
    dat = hdulist[0].data
    prihdr = hdulist[0].header
    dw = prihdr['CDELT1']
    wl0 = prihdr['CRVAL1']
    wl = wl0 + dw*np.linspace(0,len(dat)-1,num=len(dat))
    
    return wl, dat
    
def collect_fits(lstart=4500,spsize=1,vmax=2,splat=0,splong=0,inc=90,limb=0.7,nspot=1,ngrid=300,nr=20,temp=5400):
    #find all fits files for a given run - phase should be the only thing that changes

    comp = which_machine()   
    fpath0 = set_output_path(comp)  
    fpath = fpath0+'T'+str(temp)+'/fits/'

    #beginning up to unknown end wavelength
    tag0 = "fits_model0.1_"+str(lstart)+"_" 
    #end wavelength to phase
    tag1 = "_"+str(vmax)+"_"+str(splat)+"_"+str(splong)+"_"+str(inc)+"_"+str(limb)+"_"
    tag2 = "_"+str(nspot)+"_"+str(ngrid)+"_"+str(nr)+".fits"
    
    search = fpath + tag0+"*"+tag1+"*"+tag2
    print search
    
    filelist = glob.glob(search)
    filelist.sort()
    return filelist
    
    
def collect_fits_quick(temp=5400):
    comp = which_machine()   
    rpath0 = set_output_path(comp)  
    rpath = rpath0+'T'+str(temp)+'/fits/'
    
    #fstart1 = rpath+"fits_model0.1_4500_4700_2_0_0_90_0.7_"
    #fstart2 = rpath+"fits_model0.1_4700_4900_2_0_0_90_0.7_"
    #fstart3 = rpath+"fits_model0.1_4900_5100_2_0_0_90_0.7_"
    #fstart4 = rpath+"fits_model0.1_5100_5300_2_0_0_90_0.7_"
    #fstart5 = rpath+"fits_model0.1_5300_5500_2_0_0_90_0.7_"
    #fstart6 = rpath+"fits_model0.1_5500_5700_2_0_0_90_0.7_"
    #fstart7 = rpath+"fits_model0.1_5700_5900_2_0_0_90_0.7_"
    #fstart8 = rpath+"fits_model0.1_5900_6100_2_0_0_90_0.7_"
    #fstart9 = rpath+"fits_model0.1_6100_6300_2_0_0_90_0.7_"
    #fstart10 = rpath+"fits_model0.1_6300_6500_2_0_0_90_0.7_"
    #fstart11 = rpath+"fits_model0.1_6500_6700_2_0_0_90_0.7_"
    #fstart12 = rpath+"fits_model0.1_6700_6800_2_0_0_90_0.7_"
    #fend = "_1_300_20.fits"
    
    ##outfilestart = "/home/sgettel/cfasgettel-bulk/spot_models/T5400/fits/fits_model0.1_4500_6800_2_0_0_90_0.7_"     
    #outfilestart = "/pool/vonnegut0/harpsn/spot_modeler_spec/T5400/fits_model0.1_4500_6800_2_0_0_90_0.7_"
    
    
   


    fstart1 = rpath+"fits_model500_0.141_4500_4600_2_0_0_90_0.7_"
    fstart2 = rpath+"fits_model500_0.141_4600_4700_2_0_0_90_0.7_"
    fstart3 = rpath+"fits_model500_0.141_4700_4800_2_0_0_90_0.7_"
    fstart4 = rpath+"fits_model500_0.141_4800_4900_2_0_0_90_0.7_"
    fstart5 = rpath+"fits_model500_0.141_4900_5000_2_0_0_90_0.7_"
    fstart6 = rpath+"fits_model500_0.141_5000_5100_2_0_0_90_0.7_"
    fstart7 = rpath+"fits_model500_0.141_5100_5200_2_0_0_90_0.7_"
    fstart8 = rpath+"fits_model500_0.141_5200_5300_2_0_0_90_0.7_"
    fstart9 = rpath+"fits_model500_0.141_5300_5400_2_0_0_90_0.7_"
    fstart10 = rpath+"fits_model500_0.141_5400_5500_2_0_0_90_0.7_"
    fstart11 = rpath+"fits_model500_0.141_5500_5600_2_0_0_90_0.7_"
    fstart12 = rpath+"fits_model500_0.141_5600_5700_2_0_0_90_0.7_"
    fstart13 = rpath+"fits_model500_0.141_5700_5800_2_0_0_90_0.7_"
    fstart14 = rpath+"fits_model500_0.141_5800_5900_2_0_0_90_0.7_"
    fstart15 = rpath+"fits_model500_0.141_5900_6000_2_0_0_90_0.7_"
    fstart16 = rpath+"fits_model500_0.141_6000_6100_2_0_0_90_0.7_"
    fstart17 = rpath+"fits_model500_0.141_6100_6200_2_0_0_90_0.7_"
    fstart18 = rpath+"fits_model500_0.141_6200_6300_2_0_0_90_0.7_"
    fstart19 = rpath+"fits_model500_0.141_6300_6400_2_0_0_90_0.7_"
    fstart20 = rpath+"fits_model500_0.141_6400_6500_2_0_0_90_0.7_"
    fstart21 = rpath+"fits_model500_0.141_6500_6600_2_0_0_90_0.7_"
    fstart22 = rpath+"fits_model500_0.141_6600_6700_2_0_0_90_0.7_"
    fstart23 = rpath+"fits_model500_0.141_6700_6800_2_0_0_90_0.7_"
    
    fend = "_1_300_20.fits"
    
   
    outfilestart = "/pool/vonnegut0/harpsn/spot_modeler_spec/T5400/fits_model500_0.141_4500_6800_2_0_0_90_0.7_"

    
    phase_start=-180.0
    phase_step=15
    #phases = [0.,30.]    
    
    for ph in range(24):
    #for ph in range(len(phases)):
        phase = phase_start + ph*phase_step
        #phase = phases[ph]
        hdulist1 = fits.open(fstart1+str(phase)+fend)
        hdulist2 = fits.open(fstart2+str(phase)+fend)
        hdulist3 = fits.open(fstart3+str(phase)+fend)
        hdulist4 = fits.open(fstart4+str(phase)+fend)
        hdulist5 = fits.open(fstart5+str(phase)+fend)
        hdulist6 = fits.open(fstart6+str(phase)+fend)
        hdulist7 = fits.open(fstart7+str(phase)+fend)
        hdulist8 = fits.open(fstart8+str(phase)+fend)
        hdulist9 = fits.open(fstart9+str(phase)+fend)
        hdulist10 = fits.open(fstart10+str(phase)+fend)
        hdulist11 = fits.open(fstart11+str(phase)+fend)
        hdulist12 = fits.open(fstart12+str(phase)+fend)
        hdulist13 = fits.open(fstart13+str(phase)+fend)
        hdulist14 = fits.open(fstart14+str(phase)+fend)
        hdulist15 = fits.open(fstart15+str(phase)+fend)
        hdulist16 = fits.open(fstart16+str(phase)+fend)
        hdulist17 = fits.open(fstart17+str(phase)+fend)
        hdulist18 = fits.open(fstart18+str(phase)+fend)
        hdulist19 = fits.open(fstart19+str(phase)+fend)
        hdulist20 = fits.open(fstart20+str(phase)+fend)
        hdulist21 = fits.open(fstart21+str(phase)+fend)
        hdulist22 = fits.open(fstart22+str(phase)+fend)
        hdulist23 = fits.open(fstart23+str(phase)+fend)
        
    
        header = hdulist1[0].header
        keys0 = header.keys()        
        
        dat1 = hdulist1[0].data
        dat2 = hdulist2[0].data
        dat3 = hdulist3[0].data
        dat4 = hdulist4[0].data
        dat5 = hdulist5[0].data
        dat6 = hdulist6[0].data
        dat7 = hdulist7[0].data
        dat8 = hdulist8[0].data
        dat9 = hdulist9[0].data
        dat10 = hdulist10[0].data
        dat11 = hdulist11[0].data
        dat12 = hdulist12[0].data
        dat13 = hdulist13[0].data
        dat14 = hdulist14[0].data
        dat15 = hdulist15[0].data
        dat16 = hdulist16[0].data
        dat17 = hdulist17[0].data
        dat18 = hdulist18[0].data
        dat19 = hdulist19[0].data
        dat20 = hdulist20[0].data
        dat21 = hdulist21[0].data
        dat22 = hdulist22[0].data
        dat23 = hdulist23[0].data
        
        #print len(dat1), len(dat2), len(dat3), len(dat4), len(dat5), len(dat6)
    
    
        datall = np.append(dat1,dat2)
        datall = np.append(datall,dat3)
        datall = np.append(datall,dat4)
        datall = np.append(datall,dat5)
        datall = np.append(datall,dat6)
        datall = np.append(datall,dat7)
        datall = np.append(datall,dat8)
        datall = np.append(datall,dat9)
        datall = np.append(datall,dat10)
        datall = np.append(datall,dat11)
        datall = np.append(datall,dat12)
        datall = np.append(datall,dat13)
        datall = np.append(datall,dat14)
        datall = np.append(datall,dat15)
        datall = np.append(datall,dat16)
        datall = np.append(datall,dat17)
        datall = np.append(datall,dat18)
        datall = np.append(datall,dat19)
        datall = np.append(datall,dat20)
        datall = np.append(datall,dat21)
        datall = np.append(datall,dat22)
        datall = np.append(datall,dat23)
        
        #print len(datall)
        
        hdu = fits.PrimaryHDU(datall)
        hdulist = fits.HDUList([hdu])
        prihdr = hdulist[0].header
        
        for key in keys0:
            #print key, header
            prihdr[key] = header[key]
            
        print header["PHASE"]
        prihdr["NAXIS1"] = len(datall)
    
        print 'writing ',outfilestart+str(phase)+'_1_300_20.fits'
        hdulist.writeto(outfilestart+str(phase)+'_1_300_20.fits',clobber=True)
    


