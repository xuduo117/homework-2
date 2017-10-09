#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:52:41 2017

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
image_size=150
#parameter_all=np.loadtxt('fitting_gaussian.txt')
#PA_all=np.loadtxt('PA.txt')
parameter_all=np.loadtxt('fitting_gaussian_2.txt')
PA_all=np.loadtxt('PA_2.txt')
x_cen_all=parameter_all[:,2]
y_cen_all=parameter_all[:,1]

r_plot=np.arange(0,300)
#save_path='../newfits/ROXs42B/'
save_path='../newfits/ROXs12/'

image_all=np.zeros([len(y_cen_all),1024,1024])

for ctt in range(0,len(y_cen_all)):
#for ctt in range(0,1):
    brightness_all=[]
    hdulist = fits.open(save_path+str(ctt)+'_shift.fits')
    image=hdulist[0].data
    for r in r_plot:
        mask=ring_mask(611,470,r,1)
        brightness=median_value_ring(image,mask)
        brightness_all.append(brightness)
        image[mask]=image[mask]-brightness
        
        
        
    plt.figure(1)
    plt.clf()
    plt.plot(r_plot,np.asarray(brightness_all))
    #plt.xlim([590,630])
    #plt.ylim([451,491])
    plt.show()
#    plt.savefig('../image/3/'+str(ctt)+'_profile.png')
    plt.savefig('../image/3_2/'+str(ctt)+'_profile.png')

    plt.figure(2)
    plt.clf()
    plt.imshow(image,origin='lower')
#    plt.xlim([611-image_size,611+image_size])
#    plt.ylim([470-image_size,470+image_size])
    plt.show()
#    plt.savefig('../image/3/'+str(ctt)+'_subtract_shift.png')
    plt.savefig('../image/3_2/'+str(ctt)+'_subtract_shift.png')
    
    plt.figure(3)
    plt.clf()
    plt.imshow(image,origin='lower')
    plt.xlim([611-image_size,611+image_size])
    plt.ylim([470-image_size,470+image_size])
    plt.show()
#    plt.savefig('../image/3/'+str(ctt)+'_subtract_shift_crop.png')
    plt.savefig('../image/3_2/'+str(ctt)+'_subtract_shift_crop.png')
    fits.writeto(save_path+str(ctt)+'_subtract_shift.fits',image,overwrite=True)
    
    
    image_all[ctt,:,:]=image


#    
##    plt.savefig('../image/2/'+str(ctt)+'_shift.png')
#    
#


#fits.writeto('shift_sum.fits',np.sum(image_all,axis=0),overwrite=True)
#fits.writeto('shift_median.fits',np.median(image_all,axis=0),overwrite=True)














