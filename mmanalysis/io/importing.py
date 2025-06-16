# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:41:39 2023

@author: Tim Kodalle
"""
import numpy as np
import pandas as pd
import os
import glob
from scipy.signal import savgol_filter
import copy
from datetime import datetime, timedelta
from tqdm import tqdm
import mMA_pyFAICalibration
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

def loadData(filePath, calibration):
    # Open the HDF5 file
    with h5py.File(filePath, 'r') as h5_file:
        
        dataset = h5_file['Data']
        
        #%% GIWAXS-Data
            
        # List all datasets in the 'images' group
        giwaxsDatasets = dataset['images']
        giwaxsDatasetNames = list(giwaxsDatasets.keys())

        # Load each dataset into a NumPy array
        giwaxsData = []
        for datasetName in giwaxsDatasetNames:
            if datasetName.startswith('Pilatus'):
                giwaxsImage = giwaxsDatasets[datasetName]
                imageData = giwaxsImage[...]
                giwaxsData.append(imageData)
                
        baseImage = h5_file['base_image'][:]
    
        #%% PL-Data  
  
        # List all datasets in the 'spectrums' group
        plDatasets = dataset['spectrums']
        plDatasetNames = list(plDatasets.keys())

        # Load each dataset into a NumPy array
        plData = []
        wlData = []
        firstDataset = True
        for datasetName in plDatasetNames:
            plDataset = plDatasets[datasetName]
            data = plDataset[...]
            if firstDataset:
                wlData.append(data.transpose()[0])
                firstDataset = False
            plData.append(data.transpose()[1])

        #%% Logging-Data
            
        # List all datasets in the 'images' group
        loggingDataset = dataset['AI']
        loggingData = loggingDataset[...]
        # loggingHeader1 = loggingDataset.attrs['Data Names']
        # loggingHeader2 = loggingDataset.attrs['Images']
        # loggingHeader = np.concatenate((loggingHeader1, loggingHeader2))
        loggingHeader = loggingDataset.attrs['Data Names']
        
    return plData, wlData, giwaxsData, loggingData, loggingHeader, baseImage

def integrate_image(ai, image, npt=1000, azimuth_range=None):
    if azimuth_range is not None:
        azimuth_range = tuple(azimuth_range)
    q, intensity = ai.integrate1d(data = image, npt=npt, unit="q_A^-1", azimuth_range=azimuth_range)
    return q, intensity

def compute_average_intensity(image, roi):
    """
    Compute the average intensity of pixels within the region of interest (ROI).
    
    Args:
        image (numpy.ndarray): The input image.
        roi (numpy.ndarray): A binary mask representing the region of interest.
        
    Returns:
        float: The average intensity of pixels within the ROI.
    """
    # Sum the pixel values in the region of interest (this is much faster than indexing and using np.mean)
    sum_intensity = np.sum(image * roi)
    
    # Count the number of pixels in the region of interest
    count_pixels = np.sum(roi)
    
    # Compute the average intensity
    average_intensity = sum_intensity / count_pixels
    
    return average_intensity

def create_roi(image_shape, roi_coordinates):
    """
    Create a binary mask representing the region of interest (ROI).
    
    Args:
        image_shape (tuple): The shape of the image (height, width).
        roi_coordinates (tuple): The coordinates of the ROI (start_y, start_x, end_y, end_x).
        
    Returns:
        numpy.ndarray: A binary mask representing the ROI.
    """
    # Create an empty binary mask
    roi = np.zeros(image_shape, dtype=np.uint8)
    
    # Set the ROI region to 1
    for i in range(0, len(roi_coordinates)):
        start_y, start_x, end_y, end_x = roi_coordinates[i]
        roi[start_y:end_y, start_x:end_x] = 1
    
    return roi

def convertPL(wavelength, time, data, plParams):

    time = time.squeeze()
    
    data = np.transpose(data - np.mean(data[-1,:])) 

    if plParams['smoothing']:
        data = savgol_filter(data, plParams['sFactor'], 0)
        
    #Option to sum up a certain number (binning) of spectra each to improve the fitting accuracy in trade for a loss of time resolution.
    if plParams['binning'] > 0:
        data_Bin = copy.deepcopy(data)
        data_Bin = data[:, 0:int(time.shape[0]/plParams['binning'])]
        for i in range(0, data_Bin.shape[1]):
            data_Bin[:,i] = np.sum(data[:, plParams['binning']*i:plParams['binning']*i+plParams['binning']], axis=1)
        data = data_Bin
        time = time[::plParams['binning']][0:data_Bin.shape[1]]

    # transition to energy scale of the y axis
    
    energy = [1240 / i for i in wavelength]

    # Jacobian transformation for all measured PL values (basically dividing by E^2)
    data_E = data.copy()
    for i in range(np.shape(data)[1]):
        data_E[:, i] = data[:, i] / energy / energy

    # Mirroring dataframes to prevent sorting issues  ## check if still correct
    energy = np.flip(energy).squeeze()
    data_E = np.flip(data_E, axis=0)
    
    # Making a log-version of the intensity dataframe for contour plots
    data_Elog = copy.deepcopy(data_E)
    data_Elog = np.where(data_Elog < 0.1, 0.1, data_Elog)
    data_Elog = np.log(data_Elog)
    
    return time, wavelength, energy, data_E, data_Elog 

def getData(giwaxsParams, plParams, sampleNames, calib_file_path, outputPath, h5_files):
    
    # Load the calibration file
    ai = AzimuthalIntegrator()
    ai.load(calib_file_path)

    # User-defined integration parameters
    npt = giwaxsParams['ai_npts']
    azimuth_range = giwaxsParams['ai_range']
    loggingData_batch = []
    q_values_batch = []
    giwaxsTime_batch = []
    giwaxsData1D_batch = []
    plTimePre_batch = []
    plWavelengthPre_batch = []
    plEnergyPre_batch = []
    plDataPre_batch = []
    plDataLogPre_batch = []


    # Process each file
    for i in range(0, len(h5_files)):
        plData, wlData, giwaxsData2D, loggingData, loggingHeader, recalib_image = loadData(h5_files[i], None)
        giwaxsData1D = []
        q_values = []
        
        # Create and simplify logging dataframe:
        dfLoggingComplete = pd.DataFrame(loggingData, columns=loggingHeader)   
        dfLogging = dfLoggingComplete[["Time (s)", "Pilatus Counts", "QEPro Counts", "Pyrometer", "Spin Motor", "Dispense X", "Gas Quenching"]]
        
        # Create time arrays for PL and GIWAXS:
        uniquePL, firstIdxPL = np.unique(dfLogging["QEPro Counts"].to_numpy(), return_index=True)
        plTime = dfLogging["Time (s)"].to_numpy()[firstIdxPL]
        uniqueGIWAXS, firstIdxGIWAXS = np.unique(dfLogging.iloc[:, dfLogging.columns.to_list().index('Pilatus Counts')].to_numpy(), return_index=True)
        giwaxsTime = dfLogging["Time (s)"].to_numpy()[firstIdxGIWAXS]
        
        # Re-calibration using the substrate of the actual sample
        if giwaxsParams['Re-calibrant'] == 'ITO':
            recalib_calibrant_file = giwaxsParams['ITO-calibrant']
            newPONIPath = outputPath + '/' + sampleNames[i] + '_ITO-calib.poni'
            print("Refining calibration for Sample " + sampleNames[i] + '...')
            # Perform substrate-recalibration
            mMA_pyFAICalibration.refine_calibration(sampleNames[i], recalib_image.T, calib_file_path, recalib_calibrant_file, newPONIPath)
            ai.load(newPONIPath)
            
        else:
            print("Re-calibrant not found, using default calibration. Please update mMA_settings.py")
            
        # Background/I-Zero correction
            
        # Define the coordinates of the region of interest for background correction (start_y, start_x, end_y, end_x)
        roi_coordinates = [(1, 1, 190, 125)]
        # Create the region of interest (ROI) mask
        roi_mask = create_roi(giwaxsData2D[-1].shape, roi_coordinates)
        # # Plot the image with the ROI marked
        # plt.figure(figsize=(8, 6))
        # plt.imshow(roi_mask, cmap='gray')  # Overlay the ROI mask
        # plt.imshow(giwaxsData2D[-1], cmap='jet', norm=colors.LogNorm(vmin=0.1, vmax=np.max(giwaxsData2D[-1])/10), alpha=0.5)
        # plt.title('Image with ROI')
        # plt.colorbar(label='Intensity')
        # plt.show()
        # plt.pause(1)
        
        tempIntensity = []
            
        # GIWAXS integration
        for ii in tqdm(range(0, len(giwaxsData2D)), desc="Integrating GIWAXS frames of Sample " + str(i+1) + " of " + str(len(h5_files))):
            
            # Correcting the image intensities by an average background intensity
            tempiZero = compute_average_intensity(giwaxsData2D[ii], roi_mask)
            # Normalize intensity by background value
            # giwaxsData2D[ii] = giwaxsData2D[ii] / tempiZero ##need to figure out whats going on here once i get the updated h5 file
            
            # Perform azimuthal integration
            q, intensity = integrate_image(ai, giwaxsData2D[ii], npt=npt, azimuth_range=azimuth_range)
            giwaxsData1D.append(intensity)
            
            if ii == 0:
                q_values = q
                
        ##Just for troubleshooting:
                
        #     tempIntensity.append(tempiZero)
            
        # plt.figure(figsize=(8, 6))
        # plt.plot(giwaxsTime, tempIntensity)
        # plt.show()
        # plt.pause(1)
                
        # PL data manipulation        
        plTimePre, plWavelengthPre, plEnergyPre, plDataPre, plDataLogPre = convertPL(wlData, plTime, np.array(plData), plParams) #make sure dimensions of the array are correct for the function ## add wavelength and isolate time-array form dataframe
        
        # Collecting everything for each sample
        loggingData_batch.append(dfLogging)      
        q_values_batch.append(q_values)
        giwaxsTime_batch.append(giwaxsTime)
        giwaxsData1D_batch.append(np.array(giwaxsData1D))
        plTimePre_batch.append(plTimePre)
        plWavelengthPre_batch.append(plWavelengthPre)
        plEnergyPre_batch.append(plEnergyPre)
        plDataPre_batch.append(plDataPre)
        plDataLogPre_batch.append(plDataLogPre)
        
                
    return loggingData_batch, q_values_batch, giwaxsTime_batch, giwaxsData1D_batch, plTimePre_batch, plEnergyPre_batch, plDataPre_batch, plDataLogPre_batch


        logData = pd.read_csv(logFile, sep='\t', header = 0, names = names, skiprows = header)
        logSelection = ['Time', 'Pyrometer', 'Spin_Motor', 'Dispense X']
        logDataSelect = logData[logSelection]
        
    return logDataSelect
