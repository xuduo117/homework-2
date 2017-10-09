#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 01:50:10 2017

@author: xuduo
"""


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:46:35 2017

@author: xuduo
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy  
from astropy.io import fits
import glob, os
import time

def ring_mask(x_cen,y_cen,radius,delta_r):
    y,x = np.ogrid[-y_cen:1024-y_cen, -x_cen:1024-x_cen]
    mask = (x*x + y*y >= radius*radius) &(x*x + y*y < (radius+delta_r)**2)
    return mask

def median_value_ring(data,mask):
    n_pixel=np.sum(mask)    
    median_value=np.median(data[mask])
    return median_value

#611,470
#image_size=150
image_size=200
#parameter_all=np.loadtxt('fitting_gaussian.txt')
#PA_all=np.loadtxt('PA.txt')
parameter_all=np.loadtxt('fitting_gaussian_2.txt')
PA_all=np.loadtxt('PA_2.txt')
x_cen_all=parameter_all[:,2]
y_cen_all=parameter_all[:,1]

r_plot=np.arange(0,300)
#save_path='../newfits/ROXs42B/'
save_path='../newfits/ROXs12/'

#shift_median = fits.open('shift_median.fits')
shift_median = fits.open('shift_median_12.fits')
shift_median=shift_median[0].data
#image_all=np.zeros([len(y_cen_all),1024,1024])

for ctt in range(0,len(y_cen_all)):
#for ctt in range(0,1):
    hdulist = fits.open(save_path+str(ctt)+'_shift.fits')
    image=hdulist[0].data
    
    mask=ring_mask(611,470,13,20)
    rescale_factor=median_value_ring(image,mask)/median_value_ring(shift_median,mask)
    image_subtract_psf=image-shift_median*rescale_factor
    
        
        
    array = np.ones((1024, 1024))
    array[mask] = 0

    plt.figure(2)
    plt.clf()
    plt.imshow(image,origin='lower')
    plt.imshow(array*100,origin='lower',alpha=0.3)
    plt.xlim([611-image_size,611+image_size])
    plt.ylim([470-image_size,470+image_size])
    plt.show()
#    plt.savefig('../image/4/'+str(ctt)+'_ring_mask.png')
    plt.savefig('../image/4_2/'+str(ctt)+'_ring_mask.png')
    
    plt.figure(3)
    plt.clf()
    plt.imshow(image_subtract_psf,origin='lower')
#    plt.xlim([611-image_size,611+image_size])
#    plt.ylim([470-image_size,470+image_size])
    plt.show()
#    plt.savefig('../image/4/'+str(ctt)+'_subtract_Q6_shift.png')    
    plt.savefig('../image/4_2/'+str(ctt)+'_subtract_Q6_shift.png')    

    plt.figure(4)
    plt.clf()
    plt.imshow(image_subtract_psf,origin='lower')
    plt.xlim([611-image_size,611+image_size])
    plt.ylim([470-image_size,470+image_size])
    plt.show()
#    plt.savefig('../image/4/'+str(ctt)+'_subtract_Q6_shift_crop.png')
    plt.savefig('../image/4_2/'+str(ctt)+'_subtract_Q6_shift_crop.png')
    fits.writeto(save_path+str(ctt)+'_subtract_Q6_shift.fits',image_subtract_psf,overwrite=True)

    
#    image_all[ctt,:,:]=image


#    
##    plt.savefig('../image/2/'+str(ctt)+'_shift.png')
#    
#


#fits.writeto('shift_sum.fits',np.sum(image_all,axis=0))
#fits.writeto('shift_median.fits',np.median(image_all,axis=0))























