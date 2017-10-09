#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 00:43:30 2017

@author: xuduo
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy  
from astropy.io import fits
import glob, os
import time


def rotateImage(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = scipy.misc.imrotate(imgP, angle)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]


#"""
a, b =611,470
r = 10
#parameter_all=np.loadtxt('fitting_gaussian.txt')
#PA_all=np.loadtxt('PA.txt')
parameter_all=np.loadtxt('fitting_gaussian_2.txt')
PA_all=np.loadtxt('PA_2.txt')
x_cen_all=parameter_all[:,2]
y_cen_all=parameter_all[:,1]

#save_path='../newfits/ROXs42B/'
save_path='../newfits/ROXs12/'

file_name=[]
#for file in glob.glob("../data/ROXs42B/*.fits"):
for file in glob.glob("../data/ROXs12/*.fits"):
    file_name.append(file)

file_name=sorted(file_name)

for ctt in range(0,len(file_name)):
#for ctt,ctt1 in zip(range(5),[0,3,4,5,6]):
#for ctt in range(0,1):
    hdulist = fits.open(file_name[ctt])
    PA=hdulist[0].header['PARANG']+hdulist[0].header['ROTPPOSN']-\
    hdulist[0].header['EL']-hdulist[0].header['INSTANGL']
    delta_x=int(round(a-x_cen_all[ctt]))
    delta_y=int(round(b-y_cen_all[ctt]))
    image=hdulist[0].data
    
    image_shift=np.roll(image, delta_x, axis=0)
    image_shift=np.roll(image_shift, delta_y, axis=1)
    
    fits.writeto(save_path+str(ctt)+'_shift.fits',image_shift,overwrite=True)
    
#    plt.imshow(bb)
    image_rot=rotateImage(image_shift, -PA, [a,b])
    fits.writeto(save_path+str(ctt)+'_rot.fits',image_rot,overwrite=True)
    
    x_size=hdulist[0].header['NAXIS1']
    y_size=hdulist[0].header['NAXIS2']
    
    
    plt.figure(1)
    plt.clf()
    plt.imshow(image,origin='lower')
    plt.xlim([590,630])
    plt.ylim([451,491])
#    plt.savefig('../image/2/'+str(ctt)+'_raw.png')
    plt.savefig('../image/2_2/'+str(ctt)+'_raw_12.png')
    
    
    plt.figure(1)
    plt.clf()
    plt.imshow(image_shift,origin='lower')
    plt.xlim([590,630])
    plt.ylim([451,491])
#    plt.savefig('../image/2/'+str(ctt)+'_shift.png')
    plt.savefig('../image/2_2/'+str(ctt)+'_shift_12.png')
    
    
    
    
    plt.figure(2)
    plt.clf()
    plt.imshow(image_rot,origin='lower')
    plt.xlim([590,630])
    plt.ylim([451,491])
#    plt.savefig('../image/2/'+str(ctt)+'_rot.png')
    plt.savefig('../image/2_2/'+str(ctt)+'_rot_12.png')
    
    
    
    
    
#    y,x = np.ogrid[ -b:y_size-b,-a:x_size-a]
#    mask = x*x + y*y >= r*r
    
#    array = np.ones((y_size, x_size))
#    array[mask] = 0
#    plt.figure(2)
#    plt.clf()
#    plt.imshow(image,origin='lower')    
#    plt.imshow(array*100,origin='lower',alpha=0.3)
#    plt.xlim([590,630])
#    plt.ylim([451,491])
#    plt.title(str(ctt))
#    plt.savefig('../image/1/'+str(ctt)+'_stack.png')
    
#    image_tofit=image+0.0
#    image_tofit[mask]=0
#    plt.figure(3)
#    plt.clf()
#    plt.imshow(image_tofit,origin='lower')
#    plt.xlim([590,630])
#    plt.ylim([451,491])
#    params = fitgaussian(image_tofit)
#    fit = gaussian(*params)    
#    plt.contour(fit(*np.indices(image_tofit.shape)),cmap=plt.cm.copper)
#    plt.contour(fit(*np.indices(image_tofit.shape)),
#                levels=[params[0]/2.0], linestyles='dashed',colors='white')
#    plt.savefig('../image/1/'+str(ctt)+'_mask.png')
#    parameter_all.append(params)
    

    



