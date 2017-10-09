#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 11:56:00 2017

@author: xuduo
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy  
from astropy.io import fits
import glob, os
import time

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


def rotateImage(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = scipy.misc.imrotate(imgP, angle)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

a, b =611,470
r = 8.0
parameter_all=[]
PA_all=[]

file_name=[]
#for file in glob.glob("../data/ROXs42B/*.fits"):
for file in glob.glob("../data/ROXs12/*.fits"):
    file_name.append(file)

file_name=sorted(file_name)

for ctt in range(0,len(file_name)):
#for ctt in [0,3,4,5,6]:
#for ctt in range(0,1):
    hdulist = fits.open(file_name[ctt])
    PA=hdulist[0].header['PARANG']+hdulist[0].header['ROTPPOSN']-\
    hdulist[0].header['EL']-hdulist[0].header['INSTANGL']
    PA_all.append(PA)
    
    x_size=hdulist[0].header['NAXIS1']
    y_size=hdulist[0].header['NAXIS2']
    
    image=hdulist[0].data
    plt.figure(1)
    plt.clf()
    plt.imshow(image,origin='lower')
    plt.xlim([590,630])
    plt.ylim([451,491])
#    plt.savefig('../image/1/'+str(ctt)+'_raw.png')
    plt.savefig('../image/1_2/'+str(ctt)+'_raw.png')
    
    y,x = np.ogrid[ -b:y_size-b,-a:x_size-a]
    mask = x*x + y*y >= r*r
    
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
    
    image_tofit=image+0.0
    image_tofit[mask]=0
    plt.figure(3)
    plt.clf()
    plt.imshow(image_tofit,origin='lower')
    plt.xlim([590,630])
    plt.ylim([451,491])
    params = fitgaussian(image_tofit)
    fit = gaussian(*params)    
    plt.contour(fit(*np.indices(image_tofit.shape)),cmap=plt.cm.copper)
    plt.contour(fit(*np.indices(image_tofit.shape)),
                levels=[params[0]/2.0], linestyles='dashed',colors='white')
#    plt.savefig('../image/1/'+str(ctt)+'_mask.png')
    plt.savefig('../image/1_2/'+str(ctt)+'_mask.png')
    parameter_all.append(params)
    

    
np.savetxt('fitting_gaussian_2.txt',parameter_all)
np.savetxt('PA_2.txt',PA_all)










