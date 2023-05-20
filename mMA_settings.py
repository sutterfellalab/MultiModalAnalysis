# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:56:04 2022

@author: Tim Kodalle
"""

#%%PL-Settings:
    
def generalParameters():
    
    genParams = {
                    'GIWAXS' : True,
                    'PL' : True,
                    'Logging': True,
                    'TempOld' : False,
                    
                    'LabviewPL'  : True,     # BL PL via Labview
                    }
    
    return genParams
    
def plParameters():
    
    plParams = {    
                   'Thorlabs' : False,    # If Thorlabs software is used instead of OceanView
                   'smoothing' : False,   # Smoothing of the data to reduce noise
                   'Labview'  : True,     # BL PL via Labview
                   'sFactor' : 3,         # Parameter for smoothing with a SavGol-Filter
                   'bkgCorr' : False,     # Enable linear background removal. If True, the program will ask for two ranges for the removal. I recommend setting one of them at higher and the other at lower energy compared to the peaks of interest.
                   'bkgCorrPoly' : 1,     # This parameter determines the order of the polynomial fit used for background correction (0=const, 1=linear, etc.)
                   'binning' : 0,         # 0: no binning, >0: Binning of n spectra into one, i.e. reducing the time resolution for increased signal to noise ratio
                   'logplots': 0,
                   }
        
    return plParams