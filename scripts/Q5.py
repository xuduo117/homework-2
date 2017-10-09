#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 23:57:04 2017

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

def rotateImage(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = scipy.misc.imrotate(imgP, angle)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]


#"""


PA_all=np.loadtxt('PA.txt')
PA_all=np.loadtxt('PA_2.txt')

#save_path='../newfits/ROXs42B/'
save_path='../newfits/ROXs12/'
a, b =611,470

image_size=150
image_all_shift=np.zeros([len(PA_all),1024,1024])
image_all_rot=np.zeros([len(PA_all),1024,1024])


for ctt in range(0,len(PA_all)):
#for ctt,ctt1 in zip(range(5),[0,3,4,5,6]):
#for ctt in range(0,1):
    hdulist = fits.open(save_path+str(ctt)+'_subtract_shift.fits')
    PA=PA_all[ctt]
    image=hdulist[0].data
    
    image_rot=rotateImage(image, -PA, [a,b])
    fits.writeto(save_path+str(ctt)+'_rot.fits',image_rot,overwrite=True)
    
    x_size=hdulist[0].header['NAXIS1']
    y_size=hdulist[0].header['NAXIS2']
    image_all_shift[ctt,:,:]=image
    image_all_rot[ctt,:,:]=image_rot

    
    
    
    plt.figure(1)
    plt.clf()
    plt.imshow(image_rot,origin='lower')
#    plt.xlim([611-image_size,611+image_size])
#    plt.ylim([470-image_size,470+image_size])
#    plt.savefig('../image/3/'+str(ctt)+'_subtract_rot.png')
    
    
    
    
    plt.figure(2)
    plt.clf()
    plt.imshow(image_rot,origin='lower')
    plt.xlim([611-image_size,611+image_size])
    plt.ylim([470-image_size,470+image_size])
#    plt.savefig('../image/3/'+str(ctt)+'_subtract_rot_crop.png')
    
    
    
    

#fits.writeto('subtract_shift_sum.fits',np.sum(image_all_shift,axis=0),overwrite=True)
#fits.writeto('subtract_shift_median.fits',np.median(image_all_shift,axis=0),overwrite=True)
#
#fits.writeto('subtract_rot_sum.fits',np.sum(image_all_rot,axis=0),overwrite=True)
#fits.writeto('subtract_rot_median.fits',np.median(image_all_rot,axis=0),overwrite=True)


fits.writeto('subtract_shift_sum_12.fits',np.sum(image_all_shift,axis=0),overwrite=True)
fits.writeto('subtract_shift_median_12.fits',np.median(image_all_shift,axis=0),overwrite=True)

fits.writeto('subtract_rot_sum_12.fits',np.sum(image_all_rot,axis=0),overwrite=True)
fits.writeto('subtract_rot_median_12.fits',np.median(image_all_rot,axis=0),overwrite=True)


image_size=200
#file_all=['subtract_shift_sum','subtract_shift_median','subtract_rot_sum','subtract_rot_median']
file_all=['subtract_shift_sum_12','subtract_shift_median_12','subtract_rot_sum_12','subtract_rot_median_12']
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







