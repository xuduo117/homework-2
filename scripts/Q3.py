#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:32:41 2017

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

#parameter_all=np.loadtxt('fitting_gaussian.txt')
#PA_all=np.loadtxt('PA.txt')
parameter_all=np.loadtxt('fitting_gaussian_2.txt')
PA_all=np.loadtxt('PA_2.txt')
x_cen_all=parameter_all[:,2]
y_cen_all=parameter_all[:,1]

#save_path='../newfits/ROXs42B/'
save_path='../newfits/ROXs12/'

image_all=np.zeros([len(y_cen_all),1024,1024])

for ctt in range(0,len(y_cen_all)):
    hdulist = fits.open(save_path+str(ctt)+'_shift.fits')
#    hdulist = fits.open(save_path+str(ctt)+'_rot.fits')
    image=hdulist[0].data
    
    image_all[ctt,:,:]=image



plt.figure(1)
plt.clf()
plt.imshow(np.sum(image_all,axis=0),origin='lower')
#plt.xlim([590,630])
#plt.ylim([451,491])
plt.show()
#    plt.savefig('../image/2/'+str(ctt)+'_shift.png')
    

plt.figure(2)
plt.clf()
plt.imshow(np.median(image_all,axis=0),origin='lower')
#plt.xlim([590,630])
#plt.ylim([451,491])
plt.show()

#fits.writeto('shift_sum.fits',np.sum(image_all,axis=0),overwrite=True)
#fits.writeto('shift_median.fits',np.median(image_all,axis=0),overwrite=True)
#fits.writeto('rot_sum.fits',np.sum(image_all,axis=0),overwrite=True)
#fits.writeto('rot_median.fits',np.median(image_all,axis=0),overwrite=True)

fits.writeto('shift_sum_12.fits',np.sum(image_all,axis=0),overwrite=True)
fits.writeto('shift_median_12.fits',np.median(image_all,axis=0),overwrite=True)
#fits.writeto('rot_sum_12.fits',np.sum(image_all,axis=0),overwrite=True)
#fits.writeto('rot_median_12.fits',np.median(image_all,axis=0),overwrite=True)


#"""
image_size=150
#file_all=['rot_sum','rot_median','shift_median','shift_sum']
file_all=['rot_sum_12','rot_median_12','shift_median_12','shift_sum_12']
for ctt in range(4):
    image=readfits(file_all[ctt])
    plt.figure(1)
    plt.clf()
    plt.imshow(image,origin='lower')
    #plt.xlim([590,630])
    #plt.ylim([451,491])
    plt.show()
#    plt.savefig(file_all[ctt]+'_42B.png')
    plt.savefig(file_all[ctt]+'_12.png')
        
    
    plt.figure(2)
    plt.clf()
    plt.imshow(image,origin='lower')
    plt.xlim([611-image_size,611+image_size])
    plt.ylim([470-image_size,470+image_size])
    plt.show()
#    plt.savefig(file_all[ctt]+'_42B_crop.png')
    plt.savefig(file_all[ctt]+'_12_crop.png')
#"""










