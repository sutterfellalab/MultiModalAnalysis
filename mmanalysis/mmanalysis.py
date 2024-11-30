#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Tue Dec 20 11:12:01 2022

@author: Tim Kodalle
"""

# scientific libraries
import numpy as np
import pandas as pd

# os and pathing
import glob
import os
import ntpath

# save
import dill

# gui stuff
from tkinter import filedialog, simpledialog

# internal modules
from .visualization import plots
from .core import settings, fits
from .gui import mma_gui
from .io.importing import convertGIWAXS_data, getPLData, getLogData

#%%

class MMAnalysis(object):

    def __init__(self, name=None, restart_file=None):

        if restart_file is not None:

            print("Restarting from " + restart_file)
            self.restart_from_pickle(restart_file)

        else:
            print("Starting new analysis...")
            
            self.inputDict = {}
            
            self.genParams = settings.generalParameters()

            folder = filedialog.askdirectory()
            self.sampleName = ntpath.basename(folder)
            os.makedirs(folder + '/output', exist_ok=True)
            os.makedirs(folder + '/output/fits', exist_ok=True)
            self.outputPath = folder + '/output'

            if self.genParams['GIWAXS']:
                
                GIWAXS_file = glob.glob(folder + '/GIWAXS' + "/*.dat")[0]
                GIWAXS_data = pd.read_csv(GIWAXS_file, sep='\s+', header=0, names=np.array(
                    ['image_num', 'twotheta', 'twotheta_cuka', 'dspacing', 'qvalue', 'intensity', 'frame_number', 'izero',
                      'date', 'time', 'AM/PM']))
        
                self.qRaw, self.giwaxsTimeRaw, self.giwaxsIntensityRaw = self.convertGIWAXS_data(GIWAXS_data, self.sampleName, self.outputPath)
                
            if self.genParams['Logging']:
                if self.genParams['TempOld']:
                    logFile = glob.glob(folder + '/Logfile' + "/*.csv")[0]
                else:
                    logFile = glob.glob(folder + '/Logfile' + "/*.txt")[0]
                    
                self.logDataRaw = self.getLog_data(logFile)

            if self.genParams['PL']:

                plFiles = sorted(glob.glob(folder + '/PL' + "/*.txt"))         
                self.plTimeRaw, self.plWavelengthRaw, self.plEnergyRaw, self.plIntensityRaw, self.plIntensityERaw, self.plIntensityERawLog = self.getPL_data(plFiles, folder = folder + '/PL/')
               
            if not self.genParams['GIWAXS'] and not self.genParams['Logging'] and not self.genParams['PL']:
                print('Please select at least one data type in the settings file.')
               
        return 

    def convertGIWAXS_data(self, GIWAXS_data, sample_name, save_path):

        q, frame_time, full_intensity = convertGIWAXS_data(GIWAXS_data, sample_name, save_path)

        return (q, frame_time, full_intensity)
    
    def getPL_data(self, PL_files, folder):
        
        self.plParams = settings.plParameters()
        
        if self.genParams['Logging']:
            df_x, df_y, df_y_E, df, df_E, df_Elog  = getPLData(self.plParams, PL_files, folder, self.logDataRaw['Time'].to_numpy())
        else:
            df_x, df_y, df_y_E, df, df_E, df_Elog  = getPLData(self.plParams, PL_files, folder, [])
            
        return df_x, df_y, df_y_E, df, df_E, df_Elog 
    
    def getLog_data(self, logFile):
        
        logDataSelect = getLogData(self.genParams, logFile)
        
        return logDataSelect
    
    def plotGIWAXS(self, GIWAXS_cut, sample_name, save_path, q, frame_time, intensity):

        if GIWAXS_cut is True:
            
            #self.contourGIWAXSRaw = plots.plotGIWAXS(sample_name, save_path, q, frame_time, intensity)
            plots.plotGIWAXS(sample_name, save_path, q, frame_time, intensity)
            
            if self.genParams['LabviewPL']:
            
                mma_gui.inputGUI(self.inputDict, "Input_GIWAXS", 4, "Select Ranges", ['Please set the start time (in s):', 'Please set the end time (in s):', 'Please set the lower q (in A-1): ',
                                                               'Please set the upper q (in A-1): '], "Automated guess: " 
                          + str(round(self.suggestedGIWAXSTime,3)) + ' s and ' + str(round(float(self.inputDict["Times_Logging"][1])-float(self.inputDict["Times_Logging"][0]) + float(self.suggestedGIWAXSTime),3)) + ' s')
                
            else:
                
                mma_gui.inputGUI(self.inputDict, "Input_GIWAXS", 4, "Select Ranges", ['Please set the start time (in s):', 'Please set the end time (in s):', 'Please set the lower q (in A-1): ',
                                                               'Please set the upper q (in A-1): '], "Automated guess for the starting time is " 
                          + str(round(self.suggestedGIWAXSTime,3)) + ' s')
                      
            # Selecting the start time
            self.giwaxsTimeStartIdx = next(eStart for eStart, valStart in enumerate(frame_time) if valStart > float(self.inputDict["Input_GIWAXS"][0]))
            # Selecting the end time
            self.giwaxsTimeEndIdx = next(eStart for eStart, valStart in enumerate(frame_time) if valStart > float(self.inputDict["Input_GIWAXS"][1]))-1
            
            # Selecting the start q
            self.giwaxsQStartIdx = next(qStart for qStart, valStart in enumerate(q) if valStart > float(self.inputDict["Input_GIWAXS"][2]))
            
            # Selecting the end q
            self.giwaxsQEndIdx = next(qStart for qStart, valStart in enumerate(q) if valStart > float(self.inputDict["Input_GIWAXS"][3]))
            
            self.giwaxsTimePost = frame_time[self.giwaxsTimeStartIdx-1:self.giwaxsTimeEndIdx]
            self.giwaxsTimePost = self.giwaxsTimePost - self.giwaxsTimePost[0]
            
            self.giwaxsQPost = q[self.giwaxsQStartIdx-1:self.giwaxsQEndIdx]
            self.giwaxsIntensityPost = intensity[self.giwaxsTimeStartIdx-1:self.giwaxsTimeEndIdx, self.giwaxsQStartIdx-1:self.giwaxsQEndIdx]
            
        else:
            
            #self.contourGIWAXS = plots.plotGIWAXS(sample_name, save_path, q, frame_time, intensity)
            plots.plotGIWAXS(sample_name, save_path, q, frame_time, intensity)

        return
    
    def plotPL(self, plCut, sampleName, savePath, energy, time, intensity, wavelength, intensityWL, intensityLog):
        
        if plCut:

            #self.contourPLRaw = plots.plotPL(sampleName, savePath, energy, time, intensity)
            plots.plotPL(self.plParams, sampleName, savePath, energy, time, intensity, intensityLog)
            
            # Background subtraction
            if self.plParams['bkgCorr']:

                # Wavelength region to be used for background correction
                y_bkgStart1 = float(input('Please set the bkg-correction start (in eV): '))
                y_bkgEnd1 = float(input('Please set the bkg-correction end (in eV): '))
                y_bkgStart2 = float(input('Please set the bkg-correction start (in eV): '))
                y_bkgEnd2 = float(input('Please set the bkg-correction end (in eV): '))
                l_bkgStart1 = next(xStart for xStart, valStart in enumerate(energy) if valStart > y_bkgStart1)
                l_bkgEnd1 = next(xEnd for xEnd, valEnd in enumerate(energy) if valEnd > y_bkgEnd1) - 1
                l_bkgStart2 = next(xStart for xStart, valStart in enumerate(energy) if valStart > y_bkgStart2)
                l_bkgEnd2 = next(xEnd for xEnd, valEnd in enumerate(energy) if valEnd > y_bkgEnd2) - 1

                for i in range(0, len(time)):
                    xVals = np.concatenate([energy[l_bkgStart1:l_bkgEnd1], energy[l_bkgStart2:l_bkgEnd2]])
                    yVals = np.concatenate([intensity[l_bkgStart1:l_bkgEnd1, i], intensity[l_bkgStart2:l_bkgEnd2, i]])
                    coefs = np.polyfit(xVals, yVals, self.plParams['bkgCorrPoly'])  
                    poly1d_fn = np.poly1d(coefs)
                    intensity[:, i] = intensity[:, i] - poly1d_fn(energy[:])
            
            if self.plParams['Labview']:
            
                mma_gui.inputGUI(self.inputDict, "Input_PL", 4, "Select Ranges", ['Please set the start time (in s):', 'Please set the end time (in s):', 'Please set the lower energy threshold (in eV): ',
                                                               'Please set the upper energy threshold (in eV): '], "From Logging: " + str(self.inputDict["Times_Logging"][0]) + ' s' + " and " + str(float(self.inputDict["Times_Logging"][1])) + ' s')
            else:
                
                mma_gui.inputGUI(self.inputDict, "Input_PL", 4, "Select Ranges", ['Please set the start time (in s):', 'Please set the end time (in s):', 'Please set the lower energy threshold (in eV): ',
                                                               'Please set the upper energy threshold (in eV): '], '')
            
            
            if float(self.inputDict["Input_PL"][0]) < time[0]:
                self.inputDict["Input_PL"][0] = time[0]
                
            if float(self.inputDict["Input_PL"][1]) > time[-2]:
                 self.inputDict["Input_PL"][1] = time[-2]
                 
            if float(self.inputDict["Input_PL"][2]) < energy[0]:
                self.inputDict["Input_PL"][2] = energy[0]
               
            if float(self.inputDict["Input_PL"][3]) > energy[-2]:
                 self.inputDict["Input_PL"][3] = energy[-2]
            
            # Selecting the start time
            self.plTimeStartIdx = next(eStart for eStart, valStart in enumerate(time) if valStart > float(self.inputDict["Input_PL"][0]))
            # Selecting the end time
            self.plTimeEndIdx = next(eStart for eStart, valStart in enumerate(time) if valStart > float(self.inputDict["Input_PL"][1]))
            
            # Selecting the start enegy
            self.plEStartIdx = next(eStart for eStart, valStart in enumerate(energy) if valStart > float(self.inputDict["Input_PL"][2]))
            # Selecting the end energy
            self.plEEndIdx = next(eStart for eStart, valStart in enumerate(energy) if valStart > float(self.inputDict["Input_PL"][3]))
            
            self.plTimePost = time[self.plTimeStartIdx-1:self.plTimeEndIdx+1]
            self.plTimePost = self.plTimePost - self.plTimePost[0]
            self.plEnergyPost = energy[self.plEStartIdx-1:self.plEEndIdx+1]
            self.plIntensityPost = intensity[self.plEStartIdx-1:self.plEEndIdx+1, self.plTimeStartIdx-1:self.plTimeEndIdx+1]
            self.plIntensityLogPost = intensityLog[self.plEStartIdx-1:self.plEEndIdx+1, self.plTimeStartIdx-1:self.plTimeEndIdx+1]
            
        else:
                
            #self.contourPL = plots.plotPL(sampleName, savePath, energy, time, intensity)
            plots.plotPL(self.plParams, sampleName, savePath, energy, time, intensity, intensityLog)

        return 
    
    def plotLog(self, logCut, new, sampleName, savePath, logData):
        
        if logCut is True:
            
            plots.plotLog(sampleName, savePath, logData, new)
            
            mma_gui.inputGUI(self.inputDict, "Times_Logging", 2, "Select New Times", ['Please set the start time (in s):', 'Please set the end time (in s):'], "Automated guess for the starting time is " + str(self.logDataRaw.Time[self.suggestedLogTimeIdx-1]) + ' s')
            
            self.logTimeStartIdx = next(tStart for tStart, valStart in enumerate(logData.Time) if valStart > float(self.inputDict["Times_Logging"][0]))
            self.logTimeEndIdx = next(tStart for tStart, valStart in enumerate(logData.Time) if valStart > float(self.inputDict["Times_Logging"][1]))
            
            logDataTemp = logData.to_numpy()
            logDataTemp = logDataTemp[self.logTimeStartIdx-1:self.logTimeEndIdx+1,:] #need to take care of case where start is 0
            logDataTemp[:,0] = logDataTemp[:,0] - logDataTemp[0,0]

            self.logDataPost = pd.DataFrame(logDataTemp, columns = ['Time','Pyrometer','Spin_Motor', 'Dispense X'])
            
        else:
                
            #self.lineLog = plots.plotLog(sampleName, savePath, logData)
            plots.plotLog(sampleName, savePath, logData, new)

        return
    
    def plotStacked(self, genParams, sampleName, savePath, q, timeGIWAXS, intGIWAXS, energyPL, timePL, intPL, logData, logTimeEndIdx):
            
        #self.stackedPlot = plots.plotStacked(sampleName, savePath, q, timeGIWAXS, intGIWAXS, energyPL, timePL, intPL, logData, logTimeEndIdx)    
        plots.plotStacked(genParams, sampleName, savePath, q, timeGIWAXS, intGIWAXS, energyPL, timePL, intPL, logData, logTimeEndIdx) 

        return
    
    def plotIndividually(self, measType, types, axisDescription, fileName, sampleName, savePath, xData, timeData, yData):
        
        idxToPlot = []
        
        mma_gui.selectionGUI(self.inputDict, "Selection", "Do you want to extract individual measurements?", [
                "No",
                "Yes - just default ones",
                "Yes - I want to select specific times",
                "Yes - all " + types + " in a specific time range"])
        
        selection = self.inputDict["Selection"]

        if selection == 'No':
            return 'none'
        
        elif selection == 'Yes - just default ones':
            showEvery = int(len(timeData)/5) 
            times = range(0, len(timeData))
            idxToPlot = [i for i in times if i % showEvery == 0]
        
        elif selection == 'Yes - I want to select specific times':
            numOfSpectra = int(simpledialog.askfloat("Select " + types, "How many " + types + " do you want to extract?"))
            
            if numOfSpectra == 0:
                return 'none'
            else:
                mma_gui.inputGUI(self.inputDict, "Select_" + types, numOfSpectra, "Select " + types, [' ']*numOfSpectra, 'Select the times you want to extract (in s):')
                
            for i in range(0,len(self.inputDict["Select_" + types])):
                idx = next(tStart for tStart, valStart in enumerate(timeData) if valStart > float(self.inputDict["Select_" + types][i]))
                idxToPlot.append(idx)      
            
        elif selection == "Yes - all " + types + " in a specific time range":
            mma_gui.inputGUI(self.inputDict, "Select_Range", 2, "Select Time Range", ['Enter start time (in s): ', 'Enter end time (in s): '], 'Select the times range from which you want to extract ' + types + ':')
            startTimeIdx = next(tStart for tStart, valStart in enumerate(timeData) if valStart > float(self.inputDict["Select_Range"][0]))
            endTimeIdx = next(tEnd for tEnd, valStart in enumerate(timeData) if valStart > float(self.inputDict["Select_Range"][1]))
            idxToPlot = range(startTimeIdx-1, endTimeIdx + 1)
            
        
        intensityToPlot = []
        timesToPlot = [timeData[x] for x in idxToPlot]
        names = ['{:.2f}'.format(x) for x in timesToPlot] 
        names = [sampleName + measType + str(x) + '_s' for x in names]           
           
        for i in range(0,len(idxToPlot)):
                intensity_tmp = yData[idxToPlot[i],:] 
                intensityToPlot.append(intensity_tmp)
                
        intensityToPlot_array = np.array(intensityToPlot).T
                
        df = pd.DataFrame(intensityToPlot_array, columns=names)
        
        plots.plotIndividually(axisDescription, fileName, sampleName, savePath, xData, idxToPlot, intensityToPlot, timeData)
        
        return df
    
    def giwaxsFits(self, sampleName, savePath, q, timeGIWAXS, intGIWAXS):
        
        mma_gui.inputGUI(self.inputDict, "GIWAXS-Fits", 3, "Select Ranges", ['Which peak do you want to fit? ', 'Please set the lower q threshold (in A-1): ',
                                                           'Please set the upper q threshold (in A-1): '], " ")
        
        #peakName = str(input('Which peak do you want to fit? ' ))
        #lowQ = float(input('Please set the lower q threshold (in A-1): '))
        lowQIdx = next(qStart for qStart, valStart in enumerate(q) if valStart > float(self.inputDict["GIWAXS-Fits"][1]))-1
        #highQ = float(input('Please set the upper q threshold (in A-1): '))
        highQIdx = next(qEnd for qEnd, valEnd in enumerate(q) if valEnd > float(self.inputDict["GIWAXS-Fits"][2]))-1
        
        show_every = int(len(timeGIWAXS)/5)                # int n, shows every n'th frame with fit
        
        fits.fit_several_frames(q, timeGIWAXS, intGIWAXS, show_every, lowQIdx, highQIdx, sampleName, savePath, self.inputDict["GIWAXS-Fits"][0])
        
        return
       
    def plFits(self, energyPL, timePL, intPL, sampleName, savePath):
        
        numGauss = float(simpledialog.askfloat("Fit PL", 'How many Gaussians do you want to use? '))
        # Fit-parameters: Set the lower and upper limits as well as the estimated position of each peak (in nm).
        # From left to right, update as many integers as needed but keep the length of arrays at 5; extra values are ignored.
        peakLowerTH = [0.0]*int(numGauss)
        peakUpperTH = [0.0]*int(numGauss)     
        estPeakWidth = [0.0] *int(numGauss)
        minPeakWidth = [0.0] *int(numGauss)
        maxPeakWidth = [0.0] *int(numGauss)
        
        mma_gui.combinedGUI(self.inputDict, "PLFits_CenterGuesses","PLFits_CenterFixed?","PLFits_Propagate?", int(numGauss), "PL Fits", 
                             int(numGauss)*['Initial guess for Peak position (in eV): '], " ", int(numGauss)*["Fixed?"], int(numGauss)*["Propagate?"])

        for i in range(0, int(numGauss)):
        
            if float(self.inputDict["PLFits_CenterFixed?"][i]): 
                peakLowerTH[i] = float(self.inputDict["PLFits_CenterGuesses"][i]) - 0.005
                peakUpperTH[i] = float(self.inputDict["PLFits_CenterGuesses"][i]) + 0.005
                estPeakWidth[i] = (0.1/1.665)**2 
                minPeakWidth[i] = (0.0/1.665)**2
                maxPeakWidth[i] = (0.5/1.665)**2
            else:
                peakLowerTH[i] = float(self.inputDict["PLFits_CenterGuesses"][i]) - 0.2
                peakUpperTH[i] = float(self.inputDict["PLFits_CenterGuesses"][i]) + 0.2
                estPeakWidth[i] = (0.1/1.665)**2
                minPeakWidth[i] = (0/1.665)**2
                maxPeakWidth[i] = (1/1.665)**2
                
            if peakLowerTH[i] < energyPL[0]:
                peakLowerTH[i] = energyPL[0]
                
            if peakUpperTH[i] > energyPL[-1]:
                peakUpperTH[i] = energyPL[-1]
                
        show_every = int(len(timePL)/10)     # int n, shows every n'th frame with fit
                
        fits.plFitting(self.plParams, energyPL, timePL, intPL, show_every, numGauss, peakLowerTH, self.inputDict, peakUpperTH, estPeakWidth, minPeakWidth, maxPeakWidth, sampleName, savePath)
        
        return
        
       
    def saveHTMLs(self, genParams, timePL, energyPL, intPL, timeGIWAXS, q, intGIWAXS, logData, savePath, sampleName):
        
        if genParams['TempOld']:
            
            plots.htmlPlots(genParams, timePL, energyPL, intPL, timeGIWAXS, q, intGIWAXS, logData.Pyrometer, [], logData.Time, savePath, sampleName)
        
        else:
        
            plots.htmlPlots(genParams, timePL, energyPL, intPL, timeGIWAXS, q, intGIWAXS, logData.Pyrometer, logData.Spin_Motor, logData.Time, savePath, sampleName)
        
        return

    # save object as pkl file
    def save_object(self, oname=None, save_path=None):

        if oname is None:
            oname = self.sampleName

        if save_path is None:
            path = self.outputPath
        else:
            path = save_path

        if oname.endswith(".pkl"):
            oname =  path + "/" + oname
        else:
            oname = path + "/" + oname.split(".")[-1] + ".pkl"

        with open(oname, 'wb') as fout:
            dill.dump(self, fout)

        print("Saved everything as {}".format(oname))
        return

    # restart object from pkl file previously saved
    def restart_from_pickle(self, pkl_file):

        # open previously generated gpw file
        with open(pkl_file, "rb") as fin:
            restart = dill.load(fin)

        self.__dict__ = restart.__dict__.copy()

        return