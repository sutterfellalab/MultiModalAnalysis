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
from pathlib import Path

# save
import dill

# gui stuff
from tkinter import filedialog, simpledialog

# internal modules
from .visualization import plots
from .core import settings, fits
from .gui import mma_gui
from .io import importing, pyFAICalibration

#%%

class MMAnalysis(object):

    def __init__(self, name=None, restart_file=None, folder=None):

        if restart_file is not None:

            print("Restarting from " + restart_file)
            self.restart_from_pickle(restart_file)

        else:
            print("Starting new analysis...")
            
            self.inputDict = {}
            
            self.genParams = settings.generalParameters()
            self.giwaxsParams = settings.giwaxsParameters()
            self.plParams = settings.plParameters()
            
            if self.giwaxsParams['GIWAXS-calibration'] == None:
                
                self.giwaxsCalibFile = self.calibrateGIWAXS()
                
            else:
                
                self.giwaxsCalibFile = self.giwaxsParams['GIWAXS-calibration']
                

            h5_files = filedialog.askopenfilenames(title="Select the spin-coater run files", filetypes=[("H5 files", "*.h5")])
            self.folder = Path(h5_files[0]).parent
            self.numFiles = len(h5_files)
            
            self.sampleName = []
            for i in range(self.numFiles):
                self.sampleName.append(str(Path(h5_files[i]).stem))
            os.makedirs(str(self.folder) + '/output', exist_ok=True)
            os.makedirs(str(self.folder) + '/output/fits', exist_ok=True)
            self.outputPath = str(self.folder) + '/output'
            
            self.loggingBatch, self.qBatch, self.giwaxsTimeBatch, self.giwaxsIntensity2DBatch, self.plTimeBatch, self.plEnergyBatch, self.plIntensityBatch, self.plIntensityLogBatch = self.getMMAData(self.giwaxsCalibFile, self.outputPath, h5_files)           
            
            
            # Initilize batch-parameters to archive in class
            self.logTimeStartIdx = []
            self.logTimeEndIdx = []
            self.logDataPost = []
            self.giwaxsTimeStartIdx = []
            self.giwaxsTimeEndIdx = []
            self.giwaxsQStartIdx  = []
            self.giwaxsQEndIdx = []
            self.giwaxsTimePost = []
            self.giwaxsQPost = []
            self.giwaxsIntensityPost = []
            self.plTimeStartIdx = []
            self.plTimeEndIdx = []
            self.plEStartIdx = []
            self.plEEndIdx = []
            self.plTimePost = []
            self.plEnergyPost = []
            self.plIntensityPost = []
            self.plIntensityLogPost = []


    def calibrateGIWAXS(self):
        
        calibrationFile = pyFAICalibration.giwaxsCalibration(self.giwaxsParams)
        
        return calibrationFile
    
    def getMMAData(self, giwaxsCalibFile, outputPath, h5_files):
        
        logging, q, giwaxsTime, giwaxsData, plTime, energy, plData, plDataLog = importing.getData(self.giwaxsParams, self.plParams, self.sampleName, giwaxsCalibFile, outputPath, h5_files)
        
        return logging, q, giwaxsTime, giwaxsData, plTime, energy, plData, plDataLog


    def plotLog(self, logCut, sampleName, savePath, logData, file):
        
        if logCut is True:
            
            plots.plotLog(sampleName, savePath, logData)
            
            mma_gui.inputGUI(self.inputDict, "Times_Logging", 2, "Select New Times", ['Please set the start time (in s):', 'Please set the end time (in s):'], "Automated guess for the starting time is " + str(self.loggingBatch[file].iloc[self.suggestedLogTimeIdx-1,0]) + ' s')
            
            self.logTimeStartIdx.append(next(tStart for tStart, valStart in enumerate(logData.iloc[:,0]) if valStart > float(self.inputDict["Times_Logging"][0])))
            self.logTimeEndIdx.append(next(tStart for tStart, valStart in enumerate(logData.iloc[:,0]) if valStart > float(self.inputDict["Times_Logging"][1])))
            
            logDataTemp = logData.to_numpy()
            logDataTemp = logDataTemp[self.logTimeStartIdx[file]-1:self.logTimeEndIdx[file]+2,:] #need to take care of case where start is 0
            logDataTemp[:,0] = logDataTemp[:,0] - logDataTemp[0,0]

            self.logDataPost.append(pd.DataFrame(logDataTemp, columns = ["Time", "Pilatus", "QEPro", "Pyrometer", "Spin Motor", "Dispense X", "Gas Quenching"])) 
            
        else:
                
            plots.plotLog(sampleName, savePath, logData)

        return


    def plotGIWAXS(self, GIWAXS_cut, sample_name, save_path, q, frame_time, intensity, file):

        if GIWAXS_cut is True:
            
            plots.plotGIWAXS(sample_name, save_path, q, frame_time, intensity)
            

            mma_gui.inputGUI(self.inputDict, "Input_GIWAXS", 2, "Select Ranges", ['Please set the lower q (in A-1): ', 'Please set the upper q (in A-1): '], "")

            # Need to make the self.variables batches for multiple samples in one run

            # Selecting the start time
            self.giwaxsTimeStartIdx.append(next(eStart for eStart, valStart in enumerate(frame_time) if valStart >  float(self.inputDict["Times_Logging"][0]))-1)
            # Selecting the end time
            self.giwaxsTimeEndIdx.append(next(eStart for eStart, valStart in enumerate(frame_time) if valStart > float(self.inputDict["Times_Logging"][1]))+1)
            
            # Selecting the start q
            self.giwaxsQStartIdx.append(next(qStart for qStart, valStart in enumerate(q) if valStart > float(self.inputDict["Input_GIWAXS"][0])))            
            # Selecting the end q
            self.giwaxsQEndIdx.append(next(qStart for qStart, valStart in enumerate(q) if valStart > float(self.inputDict["Input_GIWAXS"][1])))
            
            giwaxsTimeTemp = frame_time[self.giwaxsTimeStartIdx[file]:self.giwaxsTimeEndIdx[file]]
            self.giwaxsTimePost.append(giwaxsTimeTemp - giwaxsTimeTemp[0])
            
            self.giwaxsQPost.append(q[self.giwaxsQStartIdx[file]:self.giwaxsQEndIdx[file]])
            self.giwaxsIntensityPost.append(intensity[self.giwaxsTimeStartIdx[file]:self.giwaxsTimeEndIdx[file], self.giwaxsQStartIdx[file]:self.giwaxsQEndIdx[file]])
            
        else:

            plots.plotGIWAXS(sample_name, save_path, q, frame_time, intensity)

        return
    
    
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
    
    
    def plotPL(self, plCut, sampleName, savePath, energy, time, intensity, intensityLog, file):
        
        if plCut:

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
                      
            mma_gui.inputGUI(self.inputDict, "Input_PL", 2, "Select Ranges", ['Please set the lower energy threshold (in eV): ',
                                                               'Please set the upper energy threshold (in eV): '], '')
            
                 
            if float(self.inputDict["Input_PL"][0]) < energy[0]:
                self.inputDict["Input_PL"][0] = energy[0]
               
            if float(self.inputDict["Input_PL"][1]) > energy[-2]:
                 self.inputDict["Input_PL"][1] = energy[-2]
            
            # Selecting the start time
            self.plTimeStartIdx.append(next(eStart for eStart, valStart in enumerate(time) if valStart > float(self.inputDict["Times_Logging"][0]))-1)
            # Selecting the end time
            self.plTimeEndIdx.append(next(eStart for eStart, valStart in enumerate(time) if valStart > float(self.inputDict["Times_Logging"][1]))+1)
            
            # Selecting the start energy
            self.plEStartIdx.append(next(eStart for eStart, valStart in enumerate(energy) if valStart > float(self.inputDict["Input_PL"][0])))
            # Selecting the end energy
            self.plEEndIdx.append(next(eStart for eStart, valStart in enumerate(energy) if valStart > float(self.inputDict["Input_PL"][1])))
            
            plTimeTemp = time[self.plTimeStartIdx[file]:self.plTimeEndIdx[file]]
            self.plTimePost.append(plTimeTemp - plTimeTemp[0])
            self.plEnergyPost.append(energy[self.plEStartIdx[file]:self.plEEndIdx[file]])
            self.plIntensityPost.append(intensity[self.plEStartIdx[file]:self.plEEndIdx[file], self.plTimeStartIdx[file]:self.plTimeEndIdx[file]])
            self.plIntensityLogPost.append(intensityLog[self.plEStartIdx[file]:self.plEEndIdx[file], self.plTimeStartIdx[file]:self.plTimeEndIdx[file]])
            
        else:
                
            plots.plotPL(self.plParams, sampleName, savePath, energy, time, intensity, intensityLog)

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

    
    def plotStacked(self, genParams, sampleName, savePath, q, timeGIWAXS, intGIWAXS, energyPL, timePL, intPL, logData, logTimeEndIdx):
            
        plots.plotStacked(genParams, sampleName, savePath, q, timeGIWAXS, intGIWAXS, energyPL, timePL, intPL, logData, logTimeEndIdx) 

        return

       
    def saveHTMLs(self, genParams, timePL, energyPL, intPL, timeGIWAXS, q, intGIWAXS, logData, savePath, sampleName):
        
        plots.htmlPlots(genParams, timePL, energyPL, intPL, timeGIWAXS, q, intGIWAXS, logData.iloc[:,4], logData.iloc[:,5], logData.iloc[:,0], savePath, sampleName)
        
        return


    # save object as pkl file
    def save_object(self, oname=None, save_path=None):

        if oname is None:
            oname = self.folder.stem

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
