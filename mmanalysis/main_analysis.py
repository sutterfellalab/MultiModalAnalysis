#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:19:29 2024

@author: Tim Kodalle
"""

# scientific libraries
import numpy as np
import pandas as pd

# visualization
import tkinter as tk

# path and stuff
import os

# internal modules
from mmanalysis.mmanalysis import MMAnalysis

#%%

def main(folder=None):

    mMA_Object = mMA_class.MMAnalysis("MMA-Sample")

    for file in range(0, mMA_Object.numFiles):
        
        print("Data selection for Sample " + mMA_Object.sampleName[file] + "...")
        
        #%%
        if mMA_Object.genParams['Logging']:
                    
            continueMain = False
        
            while continueMain is False:
        
                #Finding the start time automatically
                mMA_Object.suggestedLogTimeIdx = next(x for x, val in enumerate(mMA_Object.loggingBatch[file].iloc[:,4]) if val > 0) # 4 is spin motor as set in mMA_importing
                
                
                mMA_Object.plotLog(True, mMA_Object.sampleName[file], mMA_Object.outputPath, mMA_Object.loggingBatch[file], file)
            
                mMA_Object.plotLog(False, mMA_Object.sampleName[file], mMA_Object.outputPath, mMA_Object.logDataPost[file], file)
        
        
                if tk.messagebox.askquestion("test", "Continue?") == 'yes':
                    continueMain = True
                    
            mMA_Object.logDataPost[file].to_csv(mMA_Object.outputPath + '/'  + mMA_Object.sampleName[file] + '_Log-Data' + '.csv', index=0)
           
        #%%
        if mMA_Object.genParams['GIWAXS']:
        
            continueMain = False
    
            while continueMain is False:
    
                mMA_Object.plotGIWAXS(True, mMA_Object.sampleName[file], mMA_Object.outputPath, mMA_Object.qBatch[file], mMA_Object.giwaxsTimeBatch[file], mMA_Object.giwaxsIntensity2DBatch[file], file)
                
                mMA_Object.plotGIWAXS(False, mMA_Object.sampleName[file], mMA_Object.outputPath, mMA_Object.giwaxsQPost[file], mMA_Object.giwaxsTimePost[file], mMA_Object.giwaxsIntensityPost[file], file)
    
                if tk.messagebox.askquestion("test", "Continue?") == 'yes':
                    continueMain = True
                    
            fitting = True
            
            while fitting is True:
                    
                if tk.messagebox.askquestion("test", "(Re-)Start peak fitting?") == 'yes':
                    
                    mMA_Object.giwaxsFits(mMA_Object.sampleName[file], mMA_Object.outputPath, mMA_Object.giwaxsQPost[file], mMA_Object.giwaxsTimePost[file], mMA_Object.giwaxsIntensityPost[file])
                    
                else:
                    
                    fitting = False
            
            dfPatterns = mMA_Object.plotIndividually('GIWAXS_', 'patterns', 'q ($\AA$)', '_Indv_GIWAXS-Patterns', mMA_Object.sampleName[file], mMA_Object.outputPath, mMA_Object.giwaxsQPost[file], mMA_Object.giwaxsTimePost[file], mMA_Object.giwaxsIntensityPost[file])
            if type(dfPatterns) is not str:
                dfPatterns[mMA_Object.sampleName[file] + '_q_Patterns'] = mMA_Object.giwaxsQPost[file]
                dfPatterns.to_csv(os.path.join(mMA_Object.outputPath, mMA_Object.sampleName[file] + '_Indv_GIWAXS-Patterns.csv'), index=None)
                
        #%%    
        if mMA_Object.genParams['PL']:
                        
            continueMain = False
            
            while continueMain is False:
                
                mMA_Object.plotPL(True, mMA_Object.sampleName[file], mMA_Object.outputPath, mMA_Object.plEnergyBatch[file], mMA_Object.plTimeBatch[file], mMA_Object.plIntensityBatch[file], mMA_Object.plIntensityLogBatch[file], file)
                
                mMA_Object.plotPL(False, mMA_Object.sampleName[file], mMA_Object.outputPath, mMA_Object.plEnergyPost[file], mMA_Object.plTimePost[file], mMA_Object.plIntensityPost[file], mMA_Object.plIntensityLogPost[file], file)
                
                if tk.messagebox.askquestion("test", "Continue?") == 'yes':
                    continueMain = True
                    
            fitting = True
            
            while fitting is True:
                    
                if tk.messagebox.askquestion("test", "(Re-)Start peak fitting?") == 'yes':
                    
                    mMA_Object.plFits(mMA_Object.plEnergyPost[file], mMA_Object.plTimePost[file], mMA_Object.plIntensityPost[file], mMA_Object.sampleName[file], mMA_Object.outputPath)
            
                else:
                    
                    fitting = False
                    
            dfSpectra = mMA_Object.plotIndividually('PL_', 'spectra', 'Energy (eV)', '_Indv_PL-Spectra', mMA_Object.sampleName[file], mMA_Object.outputPath, mMA_Object.plEnergyPost[file], mMA_Object.plTimePost[file], mMA_Object.plIntensityPost[file].T)
            if type(dfSpectra) is not str:
                dfSpectra[mMA_Object.sampleName[file] + '_Energy_Spectra'] = mMA_Object.plEnergyPost[file]
                dfSpectra.to_csv(os.path.join(mMA_Object.outputPath, mMA_Object.sampleName[file] + '_Indv_PL-Spectra.csv'), index=None)
    
        #%%    
        
        if mMA_Object.genParams['GIWAXS'] and mMA_Object.genParams['PL'] and mMA_Object.genParams['Logging']:
            
            print("Working on the stacked plots. This may take a minute...")
            
            mMA_Object.plotStacked(mMA_Object.genParams, mMA_Object.sampleName[file], mMA_Object.outputPath, mMA_Object.giwaxsQPost[file], mMA_Object.giwaxsTimePost[file], mMA_Object.giwaxsIntensityPost[file], mMA_Object.plEnergyPost[file], mMA_Object.plTimePost[file], mMA_Object.plIntensityPost[file], mMA_Object.logDataPost[file], mMA_Object.logTimeEndIdx[file])
            
            mMA_Object.saveHTMLs(mMA_Object.genParams, mMA_Object.plTimePost[file], mMA_Object.plEnergyPost[file], mMA_Object.plIntensityPost[file], mMA_Object.giwaxsTimePost[file], mMA_Object.giwaxsQPost[file], mMA_Object.giwaxsIntensityPost[file], mMA_Object.logDataPost[file], mMA_Object.outputPath, mMA_Object.sampleName[file])
            
            print("_____________________________________________________________")
            
        elif mMA_Object.genParams['GIWAXS'] and mMA_Object.genParams['Logging']:
            
            print("Working on the stacked plots. This may take a minute...")
            
            mMA_Object.plotStacked(mMA_Object.genParams, mMA_Object.sampleName[file], mMA_Object.outputPath, mMA_Object.giwaxsQPost[file], mMA_Object.giwaxsTimePost[file], mMA_Object.giwaxsIntensityPost[file], [], [], [], mMA_Object.logDataPost[file], mMA_Object.logTimeEndIdx[file])
            
            mMA_Object.saveHTMLs(mMA_Object.genParams, [], [], [], mMA_Object.giwaxsTimePost[file], mMA_Object.giwaxsQPost[file], mMA_Object.giwaxsIntensityPost[file], mMA_Object.logDataPost[file], mMA_Object.outputPath, mMA_Object.sampleName[file])
            
            print("_____________________________________________________________")
            
        #%%
                        
        if mMA_Object.genParams['ImagePlots'] == 'Node-centered':  
            
            # Optimizing data for plots in Igor - take out if not using Igor
            mMA_Object.giwaxsTimePost[file] = np.append(mMA_Object.giwaxsTimePost[file], mMA_Object.giwaxsTimePost[file][-1]+mMA_Object.giwaxsTimePost[file][-1]-mMA_Object.giwaxsTimePost[file][-2])
            mMA_Object.giwaxsQPost[file] = np.append(mMA_Object.giwaxsQPost[file], mMA_Object.giwaxsQPost[file][-1]+mMA_Object.giwaxsQPost[file][-1]-mMA_Object.giwaxsQPost[file][-2])
            
            if mMA_Object.genParams['PL']:
                # Optimizing data for plots in Igor - take out if not using Igor
                mMA_Object.plTimePost[file] = np.append(mMA_Object.plTimePost[file], mMA_Object.plTimePost[file][-1]+mMA_Object.plTimePost[file][-1]-mMA_Object.plTimePost[file][-2])
                mMA_Object.plEnergyPost[file] = np.append(mMA_Object.plEnergyPost[file], mMA_Object.plEnergyPost[file][-1]+mMA_Object.plEnergyPost[file][-1]-mMA_Object.plEnergyPost[file][-2])
                
        np.savetxt(mMA_Object.outputPath + '/' + mMA_Object.sampleName[file] + '_GIWAXS_qValues.csv', mMA_Object.giwaxsQPost[file], delimiter=",", header = mMA_Object.sampleName[file] + '_q-Values')
        np.savetxt(mMA_Object.outputPath + '/' + mMA_Object.sampleName[file] + '_GIWAXS_Time.csv', mMA_Object.giwaxsTimePost[file], delimiter=",", header = mMA_Object.sampleName[file] + '_GIWAXS_Time')
        np.savetxt(mMA_Object.outputPath + '/' + mMA_Object.sampleName[file] + '_GIWAXS_Intensity.csv', mMA_Object.giwaxsIntensityPost[file], delimiter=",")
        
        if mMA_Object.genParams['PL']:
            np.savetxt(mMA_Object.outputPath + '/' + mMA_Object.sampleName[file] + '_PL_Energy.csv', mMA_Object.plEnergyPost[file], delimiter=",", header = mMA_Object.sampleName[file] + '_Energy') 
            np.savetxt(mMA_Object.outputPath + '/' + mMA_Object.sampleName[file] + '_PL_Time.csv', mMA_Object.plTimePost[file], delimiter=",", header = mMA_Object.sampleName[file] + '_PLTime')
            np.savetxt(mMA_Object.outputPath + '/' + mMA_Object.sampleName[file] + '_PL_Intensity.csv', mMA_Object.plIntensityPost[file].T, delimiter=",")
    
        
       
    mMA_Object.save_object()
    
if __name__ == "__main__":
    main()
