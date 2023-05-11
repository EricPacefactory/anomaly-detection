#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:51:25 2023

@author: eo
"""


#%% Imports

import cv2
import numpy as np

from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial


#%% Functions

def make_inferno_colormap():
    
    # Define interpolating points for rgb curves that make up the inferno map
    x_idx = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y_red = [0.0, 0.251, 0.572, 0.861, 0.987, 0.988]
    y_green = [0.0, 0.038, 0.146, 0.312, 0.638, 0.998]
    y_blue = [0.0, 0.403, 0.406, 0.231, 0.0349, 0.645]
    
    # Use lagrange polynomial to interpolate between points to approx. colormap
    x_samples = np.linspace(0, 1, 256)
    bgr_samples = []
    for y_idx in [y_blue, y_green, y_red]:
        lag_coeffs = lagrange(x_idx, y_idx).coef
        y_poly = Polynomial(np.flip(lag_coeffs))
        y_samples = np.clip(y_poly(x_samples), 0, 1)
        bgr_samples.append(np.uint8(np.round(255 * y_samples)))
    
    # Bundle output as [1, 256, 3] array, in bgr-order for use in cv2.LUT function
    return np.expand_dims(np.column_stack(bgr_samples), axis = 0)

def apply_cmap(image_1ch, cmap):
    
    '''
    Converts a 1D image into a color-mapped image using the provided colormap
    If the input image is not already in uint8 format, it will be scaled by 255
    and converted into uint8 prior to the color mapping
    
    The colormap must have a shape of: [1, 256, 3]
    '''
    
    is_uint8 = (image_1ch.dtype == np.uint8)
    image_1ch_uint8 = image_1ch if is_uint8 else np.uint8(np.round(255.0*image_1ch))
    image_3ch_uint8 = cv2.cvtColor(image_1ch_uint8, cv2.COLOR_GRAY2BGR)
    
    return cv2.LUT(image_3ch_uint8, cmap)
