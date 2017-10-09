#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 21:28:23 2017

@author: xuduo
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy  
from astropy.io import fits
import glob, os
import time


def readfits(file_name):
    rot_median = fits.open(file_name+'.fits')
    rot_median=rot_median[0].data
    return rot_median

def ring_mask(x_cen,y_cen,radius,delta_r):
    y,x = np.ogrid[-y_cen:1024-y_cen, -x_cen:1024-x_cen]
    mask = (x*x + y*y >= radius*radius) &(x*x + y*y < (radius+delta_r)**2)
    return mask

def median_value_ring(data,mask):
    n_pixel=np.sum(mask)    
    median_value=np.median(data[mask])
    return median_value


file_name_42B=[]
for file in glob.glob("../data/ROXs42B/*.fits"):
    file_name_42B.append(file)

file_name_42B=sorted(file_name_42B)

file_name_12=[]
for file in glob.glob("../data/ROXs12/*.fits"):
    file_name_12.append(file)

file_name_12=sorted(file_name_12)


PA_all_42B=np.loadtxt('PA.txt')
PA_all_12=np.loadtxt('PA_2.txt')

save_path_42B='../newfits/ROXs42B/'
save_path_12='../newfits/ROXs12/'

file_name_chisq_min=[]
#image_all=np.zeros([len(y_cen_all),1024,1024])

"""
for ctt in range(0,len(PA_all_42B)):
#for ctt in range(0,1):
    hdulist = fits.open(save_path_42B+str(ctt)+'_shift.fits')
#    hdulist = fits.open(save_path+str(ctt)+'_rot.fits')
    image_42B=hdulist[0].data
    
    mask=ring_mask(611,470,13,20)
    chi_sq=[]
    for ctt_2 in range(len(PA_all_12)):
        hdulist = fits.open(save_path_12+str(ctt_2)+'_shift.fits')
        image_12=hdulist[0].data
        rescale_factor=median_value_ring(image_42B,mask)/median_value_ring(image_12,mask)
        image_subtract_psf=image_42B-image_12*rescale_factor
        chi_sq.append(np.sum(image_subtract_psf*image_subtract_psf))
    
    chi_sq=np.asarray(chi_sq)    
    print np.argmin(chi_sq),np.min(chi_sq)
    file_name_chisq_min.append(file_name_12[np.argmin(chi_sq)][15:])
    hdulist = fits.open(save_path_12+str(np.argmin(chi_sq))+'_shift.fits')
    image_12=hdulist[0].data
    rescale_factor=median_value_ring(image_42B,mask)/median_value_ring(image_12,mask)
    image_subtract_psf=image_42B-image_12*rescale_factor
     
    fits.writeto(save_path_42B+str(ctt)+'_subtract_Q7_42B.fits',image_subtract_psf,overwrite=True)
"""

for ctt in range(0,len(PA_all_12)):
#for ctt in range(0,1):
    hdulist = fits.open(save_path_12+str(ctt)+'_shift.fits')
    image_12=hdulist[0].data
    
    mask=ring_mask(611,470,13,20)
    chi_sq=[]
    for ctt_2 in range(len(PA_all_42B)):
        hdulist = fits.open(save_path_42B+str(ctt_2)+'_shift.fits')
        image_42B=hdulist[0].data
        rescale_factor=median_value_ring(image_12,mask)/median_value_ring(image_42B,mask)
        image_subtract_psf=image_12-image_42B*rescale_factor
        chi_sq.append(np.sum(image_subtract_psf*image_subtract_psf))
    
    chi_sq=np.asarray(chi_sq)    
    print np.argmin(chi_sq),np.min(chi_sq)
    file_name_chisq_min.append(file_name_42B[np.argmin(chi_sq)][16:])
    hdulist = fits.open(save_path_42B+str(np.argmin(chi_sq))+'_shift.fits')
    image_42B=hdulist[0].data
    rescale_factor=median_value_ring(image_12,mask)/median_value_ring(image_42B,mask)
    image_subtract_psf=image_12-image_42B*rescale_factor
     
    fits.writeto(save_path_12+str(ctt)+'_subtract_Q7_12.fits',image_subtract_psf,overwrite=True)



















