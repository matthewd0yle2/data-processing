# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:18:37 2024

@author: Matt
"""

import numpy as np
from scipy.interpolate import interp1d

def fft_timeseries(t, y, upsample = None):
    """
    Parameters
    ----------
    t : array, length N
        time data
    y : array, length N
        variable data
    sample_rate : None or int, optional
        Resample data using cubic-spline with sample_rate number of points.
        The default is None, equivelant to sample_rate = N. 

    Returns
    -------
    Results of the DFFT. Magnitudes are normalised by the 2 / sample_rate.
    """
    if upsample is not None:
        new_t = np.linspace(t.min(), t.max(), upsample)
        new_y = interp1d(t, y, kind='cubic')(new_t)
    else:
        new_t = t
        new_y = y
        
    # Perform FFT
    fft_values = np.fft.fft(new_y)[:upsample//2] * 2 / upsample
    freq = np.fft.fftfreq(len(new_t), t[1] - t[0])[:upsample//2]
    
    # Calculate magnitudes
    magnitude = np.abs(fft_values) * 2 / upsample
    phase = np.angle(fft_values)
    
    return (freq, magnitude, phase, fft_values)