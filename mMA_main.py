# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:25:40 2022

@author: Tim Kodalle
"""

import mMA_class
import numpy as np
import os
import pandas as pd
import tkinter as tk


testObj = mMA_class.MMAnalysis("Testsample")

if testObj.genParams['Logging']:
            
    print("_____________________________________________________________")
    print("Logging Data Selection...")

    continueMain = False

    while continueMain is False:
        if testObj.genParams['TempOld']:
            testObj.plotLog(True, False, testObj.sampleName, testObj.outputPath, testObj.logDataRaw)
        
            testObj.plotLog(False, False, testObj.sampleName, testObj.outputPath, testObj.logDataPost)
        else:
            #Finding the start time automatically
            testObj.suggestedLogTimeIdx = next(x for x, val in enumerate(testObj.logDataRaw.Spin_Motor) if val > 0) 
            #print("Automated guess for the starting time is " + str(testObj.logDataRaw.Time[testObj.suggestedLogTimeIdx]) + ' s')

            testObj.plotLog(True, True, testObj.sampleName, testObj.outputPath, testObj.logDataRaw)
        
            testObj.plotLog(False, True, testObj.sampleName, testObj.outputPath, testObj.logDataPost)

        #if input('Continue? (y/n) ') == 'y':
        if tk.messagebox.askquestion("test", "Continue?") == 'yes':
            continueMain = True
            
    testObj.logDataPost.to_csv(testObj.outputPath + '/Log-Data.csv', index=0)
   

if testObj.genParams['GIWAXS']:

    continueMain = False
    
    print("_____________________________________________________________")
    print("GIWAXS Data Selection...")
    
    while continueMain is False:
        
        #Finding the start time automatically
        intInt = testObj.giwaxsIntensityRaw.sum(axis=1)
        intIntDiff = [x - z for x, z in zip(intInt[:-1], intInt[1:])]
        intInt_Idx = min(range(len(intIntDiff[0:10])), key=intIntDiff[0:10].__getitem__) 
        testObj.suggestedGIWAXSTime = (testObj.giwaxsTimeRaw[intInt_Idx] + testObj.giwaxsTimeRaw[intInt_Idx + 1]) / 2
        #print("Automated guess for the starting time is " + str(testObj.suggestedGIWAXSTime) + ' s')
    
        testObj.plotGIWAXS(True, testObj.sampleName, testObj.outputPath, testObj.qRaw, testObj.giwaxsTimeRaw, testObj.giwaxsIntensityRaw)
        
        testObj.plotGIWAXS(False, testObj.sampleName, testObj.outputPath, testObj.giwaxsQPost, testObj.giwaxsTimePost, testObj.giwaxsIntensityPost)
        
        #if input('Continue? (y/n) ') == 'y':
        if tk.messagebox.askquestion("test", "Continue?") == 'yes':
            continueMain = True
            
    np.savetxt(testObj.outputPath + '/' + testObj.sampleName + '_GIWAXS_qValues.csv', testObj.giwaxsQPost, delimiter=",", header = testObj.sampleName + '_q-Values')
    np.savetxt(testObj.outputPath + '/' + testObj.sampleName + '_GIWAXS_Time.csv', testObj.giwaxsTimePost, delimiter=",", header = testObj.sampleName + '_GIWAXS_Time')
    np.savetxt(testObj.outputPath + '/' + testObj.sampleName + '_GIWAXS_Intensity.csv', testObj.giwaxsIntensityPost, delimiter=",")
            
    fitting = True
    
    while fitting is True:
            
        #if input("(Re-)Start peak fitting? (y/n) ") == 'y':
        if tk.messagebox.askquestion("test", "(Re-)Start peak fitting?") == 'yes':
            
            testObj.giwaxsFits(testObj.sampleName, testObj.outputPath, testObj.giwaxsQPost, testObj.giwaxsTimePost, testObj.giwaxsIntensityPost)
            
        else:
            
            fitting = False
    
    dfPatterns = testObj.plotIndividually('GIWAXS_', 'patterns', 'q ($\AA$)', '_Indv_GIWAXS-Patterns', testObj.sampleName, testObj.outputPath, testObj.giwaxsQPost, testObj.giwaxsTimePost, testObj.giwaxsIntensityPost)
    if type(dfPatterns) is not str:
        dfPatterns[testObj.sampleName + '_q_Patterns'] = testObj.giwaxsQPost
        dfPatterns.to_csv(os.path.join(testObj.outputPath, testObj.sampleName + '_Indv_GIWAXS-Patterns.csv'), index=None)
    
if testObj.genParams['PL']:
        
    print("_____________________________________________________________")
    print("PL Data Selection...")
    
    continueMain = False
    
    while continueMain is False:
        
        testObj.plotPL(True, testObj.sampleName, testObj.outputPath, testObj.plEnergyRaw, testObj.plTimeRaw, testObj.plIntensityERaw, testObj.plWavelengthRaw, testObj.plIntensityRaw, testObj.plIntensityERawLog)
        
        testObj.plotPL(False, testObj.sampleName, testObj.outputPath, testObj.plEnergyPost, testObj.plTimePost, testObj.plIntensityPost, testObj.plWavelengthRaw, testObj.plIntensityRaw, testObj.plIntensityLogPost)
        
        #if input('Continue? (y/n) ') == 'y':
        if tk.messagebox.askquestion("test", "Continue?") == 'yes':
            continueMain = True
         
    # Optimizing data for plots in Igor - take out if not using Igor
    #df_Igor = np.transpose(df_cut)
    testObj.plTimePostIgor = np.append(testObj.plTimePost, testObj.plTimePost[-1]+testObj.plTimePost[-1]-testObj.plTimePost[-2])
    testObj.plEnergyPostIgor = np.append(testObj.plEnergyPost, testObj.plEnergyPost[-1]+testObj.plEnergyPost[-1]-testObj.plEnergyPost[-2])
    
    np.savetxt(testObj.outputPath + '/' + testObj.sampleName + '_PL_Energy.csv', testObj.plEnergyPostIgor, delimiter=",", header = testObj.sampleName + '_Energy') 
    np.savetxt(testObj.outputPath + '/' + testObj.sampleName + '_PL_Time.csv', testObj.plTimePostIgor, delimiter=",", header = testObj.sampleName + '_PLTime')
    np.savetxt(testObj.outputPath + '/' + testObj.sampleName + '_PL_Intensity.csv', testObj.plIntensityPost, delimiter=",")
            
    fitting = True
    
    while fitting is True:
            
        #if input("(Re-)Start peak fitting? (y/n) ") == 'y':
        if tk.messagebox.askquestion("test", "(Re-)Start peak fitting?") == 'yes':
            
            testObj.plFits(testObj.plEnergyPost, testObj.plTimePost, testObj.plIntensityPost, testObj.sampleName, testObj.outputPath)
    
        else:
            
            fitting = False
            
    dfSpectra = testObj.plotIndividually('PL_', 'spectra', 'Energy (eV)', '_Indv_PL-Spectra', testObj.sampleName, testObj.outputPath, testObj.plEnergyPost, testObj.plTimePost, testObj.plIntensityPost.T)
    if type(dfSpectra) is not str:
        dfSpectra[testObj.sampleName + '_Energy_Spectra'] = testObj.plEnergyPost
        dfSpectra.to_csv(os.path.join(testObj.outputPath, testObj.sampleName + '_Indv_PL-Spectra.csv'), index=None)
        
    dfSpectra2 = testObj.plotIndividually('PL_', 'spectra', 'Energy (eV)', '_Indv_PL-Spectra_2', testObj.sampleName, testObj.outputPath, testObj.plEnergyPost, testObj.plTimePost, testObj.plIntensityPost.T)
    if type(dfSpectra2) is not str:
        dfSpectra2[testObj.sampleName + '_Energy_Spectra'] = testObj.plEnergyPost
        dfSpectra2.to_csv(os.path.join(testObj.outputPath, testObj.sampleName + '_Indv_PL-Spectra_2.csv'), index=None)
        
#%%    

if testObj.genParams['GIWAXS'] and testObj.genParams['PL'] and testObj.genParams['Logging']:
    
    print("_____________________________________________________________")
    print("Working on the stacked plots. This may take a minute...")
    
    testObj.plotStacked(testObj.genParams, testObj.sampleName, testObj.outputPath, testObj.giwaxsQPost, testObj.giwaxsTimePost, testObj.giwaxsIntensityPost, testObj.plEnergyPost, testObj.plTimePost, testObj.plIntensityPost, testObj.logDataPost, testObj.logTimeEndIdx)
    
    testObj.saveHTMLs(testObj.genParams, testObj.plTimePost, testObj.plEnergyPost, testObj.plIntensityPost, testObj.giwaxsTimePost, testObj.giwaxsQPost, testObj.giwaxsIntensityPost, testObj.logDataPost, testObj.outputPath, testObj.sampleName)
    
elif testObj.genParams['GIWAXS'] and testObj.genParams['Logging']:
    
    print("_____________________________________________________________")
    print("Working on the stacked plots. This may take a minute...")
    
    testObj.plotStacked(testObj.genParams, testObj.sampleName, testObj.outputPath, testObj.giwaxsQPost, testObj.giwaxsTimePost, testObj.giwaxsIntensityPost, [], [], [], testObj.logDataPost, testObj.logTimeEndIdx)
    
    testObj.saveHTMLs(testObj.genParams, [], [], [], testObj.giwaxsTimePost, testObj.giwaxsQPost, testObj.giwaxsIntensityPost, testObj.logDataPost, testObj.outputPath, testObj.sampleName)

elif testObj.genParams['PL']:
    
    print("_____________________________________________________________")
    print("Working on the stacked plots. This may take a minute...")
    
    #testObj.plotStacked(testObj.genParams, testObj.sampleName, testObj.outputPath, testObj.giwaxsQPost, testObj.giwaxsTimePost, testObj.giwaxsIntensityPost, [], [], [], testObj.logDataPost, testObj.logTimeEndIdx)
    testObj.logDataPost = pd.DataFrame()
    testObj.logDataPost.Pyrometer = []
    testObj.logDataPost.Spin_Motor = []
    testObj.logDataPost.Time = []
    
    testObj.saveHTMLs(testObj.genParams, testObj.plTimePost, testObj.plEnergyPost, testObj.plIntensityPost, [], [], [], testObj.logDataPost, testObj.outputPath, testObj.sampleName)


testObj.save_object()