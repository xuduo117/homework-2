#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 00:54:06 2017

@author: xuduo
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy  
from astropy.io import fits
import glob, os
import time

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def readfits(file_name):
    rot_median = fits.open(file_name+'.fits')
    rot_median=rot_median[0].data
    return rot_median

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




file_name_42B=['rot_median','subtract_rot_median','subtract_psf_rot_median','subtract_psf_Q7_rot_median']
file_name_12=['rot_median_12','subtract_rot_median_12','subtract_psf_rot_median_12','subtract_psf_Q7_rot_median_12']

#A = (1, 0)
#B = (1, -1)
#
#print(angle_between(A, B))


r=10
a0,b0=611,470

a1,b1=583,648

#a1,b1=728,470
#a2,b2=651,428

#image_size=150
image_size=200

parameter_all_1=[]
parameter_all_2=[]
title=['part 4','part 5','part 6','part 7']
#"""
for ctt in range(4):
#for ctt in range(1):
    
#    image=readfits(file_name_42B[ctt])
    image=readfits(file_name_12[ctt])

    y,x = np.ogrid[ -b1:1024-b1,-a1:1024-a1]
    mask = x*x + y*y >= r*r
    
    
    image_tofit=image+0.0
    image_tofit[mask]=0
    plt.figure(3)
    plt.clf()
    plt.imshow(image,origin='lower')
    plt.xlim([a0-image_size,a0+image_size])
    plt.ylim([b0-image_size,b0+image_size])
    params = fitgaussian(image_tofit)
    fit = gaussian(*params)    
    plt.contour(fit(*np.indices(image_tofit.shape)),
                levels=[params[0]/2.0], linestyles='dashed',colors='white')
    parameter_all_1.append(params)

#    y,x = np.ogrid[ -b2:1024-b2,-a2:1024-a2]
#    mask = x*x + y*y >= r*r
#    image_tofit=image+0.0
#    image_tofit[mask]=0
#    params = fitgaussian(image_tofit)
#    fit = gaussian(*params)    
#    plt.contour(fit(*np.indices(image_tofit.shape)),
#                levels=[params[0]/2.0], linestyles='dashed',colors='white')
#
#    parameter_all_2.append(params)
    plt.title(str(title[ctt]))
#    plt.savefig('../image/5/'+str(ctt)+'_planet_42B.png')
    plt.savefig('../image/5/'+str(ctt)+'_planet_12.png')




np.savetxt('fitting_gaussian_planet_12.txt',parameter_all_1)
#np.savetxt('fitting_gaussian_planet_42B_1.txt',parameter_all_1)
#np.savetxt('fitting_gaussian_planet_42B_2.txt',parameter_all_2)
#"""


star=np.array([611,470])
planet_42B_1=np.array([728, 470])
planet_42B_2=np.array([651, 428])
planet_12=np.array([583, 648])

angle_between(planet_42B_1, star)

dis_42B_1=planet_42B_1-star
dis_42B_2=planet_42B_2-star
dis_12=planet_12-star

north_vector=(0,1)

np.linalg.norm(dis_42B_1, ord=2) 
np.linalg.norm(dis_42B_2, ord=2) 
np.linalg.norm(dis_12, ord=2) 


print angle_between(north_vector, dis_42B_1)
print angle_between(north_vector, dis_42B_2)
print angle_between(north_vector, dis_12)











