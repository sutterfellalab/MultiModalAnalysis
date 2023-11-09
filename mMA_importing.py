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

def convertGIWAXS_data(GIWAXS_data, sample_name, save_path):
    '''
    Parameters
    ----------
    csv_path : path object,
        points towards the saved csv file.
    frames : int,
        imports the total number of frames taken for the scan
    sample_name : str,
        name of the sample. Default is the name under which scan is saved.
    save_path : path object
        where the output is saved.

    Returns
    -------
    three arrays, q, frame_num and full_intensity which are one, one and two-dimensional, respectively.
    These are saved as csv files.

    '''
    print(GIWAXS_data)
    numFrames = GIWAXS_data.image_num[len(GIWAXS_data)-1] + 1

    beginTime = GIWAXS_data.time[0]
    endTime = GIWAXS_data.time[len(GIWAXS_data)-1]

    FMT = '%H:%M:%S'
    tdelta = datetime.strptime(endTime, FMT) - datetime.strptime(beginTime, FMT)
    if tdelta.days < 0:
        tdelta = timedelta(
            days=0,
            seconds=tdelta.seconds - 43200,
            microseconds=tdelta.microseconds
        )
        
    time_per_frame = tdelta.total_seconds() / numFrames
        
    print('Time per frame was calculated to: ' + str(time_per_frame) + ' s')
    
    frames_numbers = np.unique(GIWAXS_data["frame_number"])
    q_values = np.unique(GIWAXS_data["qvalue"])
    
    begin_time = datetime.strptime(beginTime, FMT)
    
    full_intensity = []
    frame_times    = []
    for frame_nr in tqdm(frames_numbers):
        
        pd_frame = GIWAXS_data[GIWAXS_data["frame_number"] == frame_nr]
        
        data = pd_frame[['intensity', 'izero']].to_numpy()
        
        new_time = datetime.strptime(pd_frame["time"].iloc[0], FMT)
        
        
        full_intensity.append(np.prod(data, axis=1))
        frame_times.append((new_time - begin_time).seconds)
    
    
    return (q_values, np.array(frame_times), np.array(full_intensity))

def getPLData(plParams, PL_files, folder, logTimes):
      
    nFiles = len(PL_files)
    
    if plParams['Labview']:
        
        for i in range(0, len(PL_files)):
            fileTemp = PL_files[i].split('/')
            fileTemp2 = fileTemp[-1].split('\\')
            fileTemp3 = fileTemp2[-1].split(' ')
            fileNum = fileTemp3[-1].split('.')[0]
            
            if len(fileNum) == 3:
                os.rename(PL_files[i], os.path.join(folder, fileTemp3[0] + ' ' + fileTemp3[1] + ' ' + fileTemp3[2] + ' ' + '00' + fileTemp3[3]))
            elif len(fileNum) == 4:
                os.rename(PL_files[i], os.path.join(folder, fileTemp3[0] + ' ' + fileTemp3[1] + ' ' + fileTemp3[2] + ' ' + '0' + fileTemp3[3]))
        
    else:
    
        #Test if need to rename
        fileNameG = PL_files[0].split('/')
        fileNameG2 = fileNameG[-1].split('\\')
        fileTempG = fileNameG2[-1].split('__')
    
        if len(fileTempG) > 1:
        
            for i in range(0, len(PL_files)):
                fileName = PL_files[i].split('/')
                fileName2 = fileName[-1].split('\\')
                fileTemp = fileName2[-1].split('__')
                
                os.rename(PL_files[i], os.path.join(folder, fileTemp[0] + '_' + fileTemp[2]))
            
    PL_files = sorted(glob.glob(folder + "*.txt")) 

    if plParams['Thorlabs']:
        # Reading timestamps and converting to time from first spectrum in seconds
        mTime = np.zeros((nFiles, 6))
        for i in range(0, nFiles):
            # Import file information
            fileName = PL_files[i].split('\\')[1]
            timeStamp = fileName.split('_')[-4:]
            mTime[i, 0:3] = timeStamp[0:3]
            mTime[i, 3] = timeStamp[-1].split('.')[0]
            mTime[i, -2] = mTime[i, 0] * 3600 + mTime[i, 1] * 60 + mTime[i, 2] + mTime[i, 3] * 0.001
            if i > 0:
                mTime[i, -1] = mTime[i, -2] - mTime[0, -2]

        # defining X-axis using the timestamps as calculated above
        df_x = mTime[:, -1]
        # reading all files
        df_all = np.concatenate([pd.read_csv(f, sep=";", header=0) for f in PL_files], axis=1)
        
    elif plParams['Labview']:
        # defining X-axis using the timestamps from the logfile
        df_x = logTimes
        # reading all files
        df_all = np.concatenate([pd.read_csv(f, sep="\t", header=1) for f in tqdm(PL_files)], axis=1)
    else:
        # Reading timestamps and converting to time from first spectrum in seconds
        mTime = np.zeros((nFiles, 6))
        for i in range(0, nFiles):
            # Import file information
            fileName = PL_files[i].split('\\')[1]
            timeStamp = fileName.split('_')[-1]
            time = timeStamp.split('.')[0]
            mTime[i, 0:4] = time.split('-')
            mTime[i, -2] = mTime[i, 0] * 3600 + mTime[i, 1] * 60 + mTime[i, 2] + mTime[i, 3] * 0.001
            if i > 0:
                mTime[i, -1] = mTime[i, -2] - mTime[0, -2]

        # defining X-axis using the timestamps as calculated above
        df_x = mTime[:, -1]
        # reading all files
        df_all = np.concatenate([pd.read_csv(f, sep="\t", header=14) for f in PL_files], axis=1)

    # separating wavelength values
    df_y = df_all[:, 0]

    # deleting every other column (all wavelength values) leaving only intensities
    df = np.delete(df_all, list(range(0, df_all.shape[1], 2)), axis=1)

    # =======================Part 2: Initial Modifications==================================
    
    if plParams['Labview']:
        df = df - np.mean(df[:,0])

    # removing negative points from data (important for log plot, also helps with scaling)
    df = np.where(df <= 0, 0, df)
    
    if plParams['smoothing']:
        df = savgol_filter(df, plParams['sFactor'], 0)
        
    #Option to sum up a certain number (binning) of spectra each to improve the fitting accuracy in trade for a loss of time resolution.
    if plParams['binning'] > 0:
        df_Bin = copy.deepcopy(df)
        df_Bin = df[:, 0:int(df_x.shape[0]/plParams['binning'])]
        for i in range(0, df_Bin.shape[1]):
            df_Bin[:,i] = np.sum(df[:, plParams['binning']*i:plParams['binning']*i+plParams['binning']], axis=1)
        df = df_Bin
        df_x = df_x[::plParams['binning']][0:df_Bin.shape[1]]

    # transition to energy scale of the y axis
    
    df_y_E = [1240 / i for i in df_y]

    # Jacobian transformation for all measured PL values (basically dividing by E^2)
    df_E = df.copy()
    for i in range(np.shape(df)[1]):
        df_E[:, i] = 1240*df[:, i] / df_y_E / df_y_E

    # Mirroring dataframes to prevent sorting issues
    df_y_E = np.flip(df_y_E)
    df_E = np.flip(df_E, axis=0)
    
    # Making a log-version of the intensity dataframe for contour plots
    df_Elog = copy.deepcopy(df_E)
    df_Elog = np.where(df_Elog < 0.1, 0.1, df_Elog)
    df_Elog = np.log(df_Elog)
    
    return df_x, df_y, df_y_E, df, df_E, df_Elog 

def getLogData(logParams, logFile):
    
    if logParams['TempOld']:
        
        header = 1 # rows to skip 

        names=np.array(['Time of Day', 'Pyrometer'])
        logData = pd.read_csv(logFile, header = 0, names = names, skiprows = header)
        time = np.zeros(len(logData.iloc[:,0]))
        for i in range(0,len(logData.iloc[:,0])):
            mTime = logData.iloc[i,0]
            tempTime = mTime.split('-')[3]
            time[i] = float(tempTime.split(':')[0]) * 3600 + float(tempTime.split(':')[1]) * 60 + float(tempTime.split(':')[2])
            if i > 0:
                time[i] = time[i] - time[0]
        time[0] = 0.0
        logData['Time'] = time
        logSelection = ['Time', 'Pyrometer']
        logDataSelect = logData[logSelection]
        
    else:
        if logParams['LabviewPL']:
            header = 93 # rows to skip
            #names=np.array(['Time of Day', 'Time', 'Image Counts', 'Pyrometer', 'Dispense X', 'Dispense Z', 'Gas Quenching', 'Sine', 'Spin_Motor', 'BK Set Amps', 'BK Set Volts', 'BK Amps', 'BK Volts', 'BK Power', '2D Image', 'Spectrometer'])
            names=np.array(['Time of Day', 'Time', 'Image Counts', 'Pyrometer', 'Dispense X', 'Dispense Z', 'Gas Quenching', 'Sine', 'Spin_Motor', 'BK Set Amps', 'BK Set Volts', 'BK Amps', 'BK Volts', 'BK Power', '2D Image', 'Spectrometer'])
            
        else:
            header = 16 # rows to skip 
            names=np.array(['Time of Day', 'Time', 'Pyrometer', 'Dispense X', 'Dispense Z', 'Gas Quenching', 'Spin_Motor', 'BK Set Amps', 'BK Set Volts', 'BK Amps', 'BK Volts', 'BK Power', 'Sine'])
            

        logData = pd.read_csv(logFile, sep='\t', header = 0, names = names, skiprows = header)
        logSelection = ['Time', 'Pyrometer', 'Spin_Motor', 'Dispense X']
        logDataSelect = logData[logSelection]
            
    return logDataSelect