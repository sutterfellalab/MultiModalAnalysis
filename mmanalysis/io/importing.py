# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:41:39 2023

@author: Tim Kodalle
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import copy
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from importlib.resources import files

from .pyFAICalibration import giwaxsCalibration, refine_calibration

def getCalibFiles():
    
    exampleAluminaImage = files('mmanalysis.data') / 'Example_Al2O3_calib_10keV_6p25_2p0_35p0_10s.tif'
    aluminaCalibrant = files('mmanalysis.data') / 'alumina.D'
    itoCalibrant = files('mmanalysis.data') / 'ito_calibrant.D'
    defaultPONI = files('mmanalysis.data') / 'default_calibration.poni'
    
    # Convert Path objects to strings
    exampleAluminaImage = str(exampleAluminaImage)
    aluminaCalibrant = str(aluminaCalibrant)
    itoCalibrant = str(itoCalibrant)
    defaultPONI = str(defaultPONI)
        
    return exampleAluminaImage, aluminaCalibrant, itoCalibrant, defaultPONI

# Function to check if a specific number exists in both arrays - can hopefully be removed once h5 counter works correctly in LabView
def imagesMatch(numbers, number2, target):
    return target in numbers and target in number2

def loadData(islogging, isgiwaxs, ispl, filePath, calibration):
    # Open the HDF5 file
    with h5py.File(filePath, 'r') as h5_file:
        
        dataset = h5_file['Data']
        giwaxsData = []
        baseImage = []
        plData = []
        wlData = []
        loggingData = []
        loggingHeader = []
        
        #%% Logging-Data
        if islogging:
            # List all datasets in the 'images' group
            loggingDataset = dataset['AI']
            loggingData = loggingDataset[...]
            loggingHeader1 = loggingDataset.attrs['Data Names']
            loggingHeader2 = loggingDataset.attrs['Images']
            loggingHeader = np.concatenate((loggingHeader1, loggingHeader2))
            
        #%% GIWAXS-Data
        if isgiwaxs:
            # List all datasets in the 'images' group
            giwaxsDatasets = dataset['images']
            giwaxsDatasetNames = list(giwaxsDatasets.keys())
            imageNums = [float(datasetName.split()[-1]) for datasetName in giwaxsDatasetNames]
            i = 0
            # Load each dataset into a NumPy array
            for datasetName in giwaxsDatasetNames:
                if datasetName.startswith('Pilatus') and imagesMatch(imageNums, loggingData[:,-1], imageNums[i]):
                    giwaxsImage = giwaxsDatasets[datasetName]
                    imageData = giwaxsImage[...]
                    giwaxsData.append(imageData)
                i += 1
                
            baseImage = h5_file['base_image'][:]
    
        #%% PL-Data  
        if ispl:
            # List all datasets in the 'spectrums' group
            plDatasets = dataset['spectrums']
            plDatasetNames = list(plDatasets.keys())
    
            # Load each dataset into a NumPy array
            firstDataset = True
            for datasetName in plDatasetNames:
                plDataset = plDatasets[datasetName]
                data = plDataset[...]
                if firstDataset:
                    wlData.append(data.transpose()[0])
                    firstDataset = False
                plData.append(data.transpose()[1])
        
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

def getData(plParams, sampleNames, islogging, isgiwaxs, ispl, reCalibrant, calib_file_path, outputPath, h5_files):

    # Load the calibration file
    ai = AzimuthalIntegrator()
    ai.load(calib_file_path)  #path to gneral calibration poni (e.g. alumina)

    # User-defined integration parameters (first two should be moved into settings/setup at some point)
    npt = 1000
    azimuth_range = [160, 180] ## Check after image rotation if this is still correct, consider larger range and/or mirroring it
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
        plData, wlData, giwaxsData2D, loggingData, loggingHeader, recalib_image = loadData(islogging, isgiwaxs, ispl, h5_files[i], None)
        giwaxsData1D = []
        q_values = []
        
        if islogging:
            # Create and simplify logging dataframe:
            dfLoggingComplete = pd.DataFrame(loggingData, columns=loggingHeader)   
            if ispl and isgiwaxs:
                dfLogging = dfLoggingComplete[["Time (s)", "Pilatus", "QEPro", "Pyrometer", "Spin Motor", "Dispense X", "Gas Quenching"]]
            elif isgiwaxs:
                dfLogging = dfLoggingComplete[["Time (s)", "Pilatus", "Pyrometer", "Spin Motor", "Dispense X", "Gas Quenching"]]
            elif ispl:
                dfLogging = dfLoggingComplete[["Time (s)", "QEPro", "Pyrometer", "Spin Motor", "Dispense X", "Gas Quenching"]]
            # plt.imshow(recalib_image)
            
            # Collecting everything
            loggingData_batch.append(dfLogging) 
        
        if isgiwaxs:
            # Create time arrays for GIWAXS:
            uniqueGIWAXS, firstIdxGIWAXS = np.unique(dfLogging.iloc[:, dfLogging.columns.to_list().index('Pilatus')].to_numpy(), return_index=True)
            giwaxsTime = dfLogging["Time (s)"].to_numpy()[firstIdxGIWAXS]

            # Re-calibration using the substrate of the actual sample
            newPONIPath = outputPath + '/' + sampleNames[i] + '_ITO-calib.poni'
            print("Refining calibration for Sample " + sampleNames[i] + '...')
            # Perform substrate-recalibration
            refine_calibration(sampleNames[i], recalib_image.T, calib_file_path, str(reCalibrant), newPONIPath)
            ai.load(newPONIPath)
            
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

            # Collecting everything
            q_values_batch.append(q_values)
            giwaxsTime_batch.append(giwaxsTime)
            giwaxsData1D_batch.append(np.array(giwaxsData1D))
            
        if ispl:
            # Create time arrays for PL:
            uniquePL, firstIdxPL = np.unique(dfLogging["QEPro"].to_numpy(), return_index=True)
            plTime = dfLogging["Time (s)"].to_numpy()[firstIdxPL]
        
            # PL data manipulation        
            plTimePre, plWavelengthPre, plEnergyPre, plDataPre, plDataLogPre = convertPL(wlData, plTime, np.array(plData), plParams) #make sure dimensions of the array are correct for the function ## add wavelength and isolate time-array form dataframe
            
            # Collecting everything 
            plTimePre_batch.append(plTimePre)
            plWavelengthPre_batch.append(plWavelengthPre)
            plEnergyPre_batch.append(plEnergyPre)
            plDataPre_batch.append(plDataPre)
            plDataLogPre_batch.append(plDataLogPre)
        
                
    return loggingData_batch, q_values_batch, giwaxsTime_batch, giwaxsData1D_batch, plTimePre_batch, plEnergyPre_batch, plDataPre_batch, plDataLogPre_batch
