# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:56:04 2022

@author: Tim Kodalle
"""
    
# def generalParameters():
    
#     genParams = {
#                     'GIWAXS' : True,
#                     'PL' : False,
#                     'Logging': True,
                    
#                     'ImagePlots' : 'Node-centered',         # Igor Pro works node-centered, i.e. the x- and y- arrays mark the nodes or corners of each pixel in the z-data matrix
#                     # 'ImagePlots' : 'Pixel-centered        # Origin works pixel-centered, i.e. the x- and y- arrays mark the center of each pixel in the z-data matrix
#                     }
    
#     return genParams

# def giwaxsParameters():
    
#     giwaxsParams = {
#                     'GIWAXS-calibration': "G:/My Drive/Code/Multimodal_Analysis/mMA-Script_h5/files/default_calibration.poni",
#                     # 'GIWAXS-calibration': None,
                    
#                     'calibrant' : 'G:/My Drive/Code/Multimodal_Analysis/mMA-Script_h5/files/alumina.D',
#                     'calibration-image' : 'G:/My Drive/Code/Multimodal_Analysis/mMA-Script_h5/files/Example_Al2O3_calib_10keV_6p25_2p0_35p0_10s.tif',
#                     'energy' : '10',
#                     'default-poni' : "G:/My Drive/Code/Multimodal_Analysis/mMA-Script_h5/files/default_calibration.poni",
#                     'Re-calibrant' : 'ITO',
#                     'ITO-calibrant' : "G:/My Drive/Code/Multimodal_Analysis/mMA-Script_h5/files/ito_calibrant.D",
#                     'ai_npts' : 1000,
#                     'ai_range': [160, 180], ## Check after image rotation if this is still correct, consider larger range and/or mirroring it
#                     }
    
#     return giwaxsParams
    
def plParameters():
    
    plParams = {    
                   'Thorlabs' : False,    # If Thorlabs software is used instead of OceanView
                   'smoothing' : False,   # Smoothing of the data to reduce noise
                   'sFactor' : 3,         # Parameter for smoothing with a SavGol-Filter
                   'bkgCorr' : False,     # Enable linear background removal. If True, the program will ask for two ranges for the removal. I recommend setting one of them at higher and the other at lower energy compared to the peaks of interest.
                   'bkgCorrPoly' : 1,     # This parameter determines the order of the polynomial fit used for background correction (0=const, 1=linear, etc.)
                   'binning' : 0,         # 0: no binning, >0: Binning of n spectra into one, i.e. reducing the time resolution for increased signal to noise ratio
                   'logplots': False,     # Set to 'True' if you want the script to display the PL-maps with log(intensity)
                   }
        
    return plParams
