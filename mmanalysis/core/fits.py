# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:54:01 2022

@author: Tim Kodalle
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import signal
from tqdm import tqdm
from lmfit.models import LinearModel, PseudoVoigtModel
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import traceback

#%%
#GIWAXS-Fitting
   
        
def fit_single_frame(lowQ, highQ, q, intensity, frame_index, frames_to_plot, sampleName, outputPath):
    
    x = q[lowQ:highQ]
    y = intensity[frame_index, lowQ:highQ]
    
    init_params = {                     # initial guess parameters
                    'amplitude' : max(y)/40,     # default: 2
                    'center' : x[np.argmax(y)], # 1 (in angstrom-1)
                    'sigma' : 0.01,      # 0.01
                    'fraction' : 0.5,    # 0.5
                    'slope' : y[-1] - y[0], 
                    'intercept' : 0    # 700
                    }
    
    # init_params = {                     # initial guess parameters
    #                 'amplitude' : max(y)/2,     # default: 2
    #                 'center' : x[np.argmax(y)], # 1 (in angstrom-1)
    #                 'sigma' : 0.3,      # 0.01
    #                 'fraction' : 0.5,    # 0.5
    #                 'slope' : (y[-1] - y[0])/(x[-1] - x[0]), 
    #                 'intercept' : y[0] - (y[-1] - y[0])/(x[-1] - x[0])*x[0]    # 700
    #                 }

    # define fitting models (so far, one peak and a background function)
    peak = PseudoVoigtModel()
    background = LinearModel()
    mod = peak + background
    # initial values
    pars = mod.make_params(amplitude = init_params['amplitude'], 
                           center = init_params['center'], 
                           sigma = init_params['sigma'], 
                           fraction = init_params['fraction'], 
                           slope = init_params['slope'], 
                           intercept = init_params['intercept'])
    # bounds
    pars.add('center', value=init_params['center'], min=q[lowQ], max=q[highQ])
    pars.add('amplitude', value=init_params['amplitude'])
    mod.set_param_hint('amplitude', min=0)
    mod.set_param_hint('center', min=q[lowQ], max=q[highQ])
    mod.set_param_hint('sigma', max=0.01)

    # determine if peak in data, promninence of 190 is chosen by hand, doesn't
    # need to be ideal for every sample
    peak_in_frame = False #initially false
    peaks = signal.find_peaks(y)[0]
    if len(peaks) > 0:

        peak_in_frame = True
        
        # fitting call
        result = mod.fit(y, pars, x=x)

        redchi = result.redchi
        dely = result.eval_uncertainty(sigma=3)
        params = []
        std_error = []

        for name, param in result.params.items():
            params.append(param.value)
            std_error.append(param.stderr) 
        if frame_index in frames_to_plot:
            plt.figure(figsize=(7, 5))
            plt.plot(x, y, 'o', label='intensity')
            plt.plot(x[peaks], y[peaks], 'r.', label='found peak')
            plt.plot(x, result.init_fit, '--', label='initial guess')
            plt.plot(x, result.best_fit, '-', label='best fit')
            plt.fill_between(x, result.best_fit-dely, result.best_fit+dely, 
                             color='#ABABAB', label='3$\sigma$ - uncertainty band')

            plt.xlabel(r'q $(\AA)$')
            plt.ylabel(r'Intensity (au)')
            # result.plot(data_kws={'markersize': 1})
            plt.legend()
            plt.title('Frame: ' + str(frame_index))
            plt.savefig(os.path.join(outputPath + '/fits/', str(sampleName) + '_GIWAXS-fit_Frame_' + str(frame_index) + '.png'), format = 'png')
            plt.show(block=False)
            plt.pause(1)
            

    elif len(peaks) == 0:
        if frame_index in frames_to_plot:
            plt.figure(figsize=(7, 5))
            plt.plot(x, y, 'o', label='intensity')
            plt.xlabel(r'q $(\AA)$')
            plt.ylabel(r'Intensity (au)')
            # result.plot(data_kws={'markersize': 1})
            plt.legend()
            plt.title('Frame: ' + str(frame_index))
            #plt.show()
        params = [None]*6
        std_error = [None]*3
        redchi = [None]
        print("No Peak Found")

    return (params, std_error, redchi, peak_in_frame)        
    
def fit_several_frames(q, time, intensity, show_every, lowQ, highQ, sampleName, outputPath, hkl):

    amplitude, unc_a = [], []
    center, unc_c = [], []
    sigma, unc_s = [], []
    fraction = []
    slope = []
    
    intercept = []
    red_chi = []
    all_params = [amplitude, center, sigma, fraction, slope, intercept]
    peak_unc = [unc_a, unc_c, unc_s]
    frames = range(0, len(time))
    frames_to_plot = [i for i in frames if i % show_every == 0]
    for frame in tqdm(frames, desc='Fitting frames'):
        params, std_error, redchi, peak_in_frame = fit_single_frame(lowQ, highQ, q, intensity, 
                                                                    frame, frames_to_plot, sampleName, outputPath)

        red_chi.append(redchi)

        for index, param in enumerate(all_params):
            param.append(params[index])
        for index, unc in enumerate(peak_unc):
            unc.append(std_error[index])
# =============================================================================
#         if peak_in_frame:
#             if std_error[0] != None:
#                 if std_error[0] < 1:
#                     # for higher efficiency, the init_params are now changed to the 
#                     # fit values for next scan. However, if the initial frame is 
#                     # wrongly identified to contain a peak, this might lead to problems
#                     init_params['amplitude'] = params[0]
#                     init_params['center'] = params[1]
#                     init_params['sigma'] = params[2]
#                     init_params['fraction'] = params[3]
#                     init_params['slope'] = params[4]
#                     init_params['intercept'] = params[5]
# =============================================================================
                
    fig, ax1 = plt.subplots(figsize=(7, 5))
    plot1, = ax1.plot(frames, center, label='center')
    ax2 = ax1.twinx()
    plot2, = ax2.plot(frames, sigma, 'g', label='$\sigma$')
    ax1.set_xlabel('Frame #')
    ax1.set_ylabel(r'q ($\AA^{-1}$)')
    ax2.set_ylabel(r' $\sigma$ ($\AA^{-1}$)')
    # Create your ticker object with M ticks
    yticks = ticker.MaxNLocator(5)
    ax1.yaxis.set_major_locator(yticks)
    fig.suptitle('Fit Results ' + sampleName, fontsize=14)
    fig.legend()
    plt.pause(1)
    
    # saving peak fit params in separate csv files
    params_to_save = {sampleName + '_' + hkl + '_time (s)' : time,
                      sampleName + '_' + hkl + '_amplitude (au)' : amplitude, 
                      sampleName + '_' + hkl + '_center ($\AA$)' : center, 
                      sampleName + '_' + hkl + '_sigma ($\AA$)' : sigma, 
                      sampleName + '_' + hkl + '_std error amplitude (au)' : unc_a, 
                      sampleName + '_' + hkl + '_std error center ($\AA$)' : unc_c, 
                      sampleName + '_' + hkl + '_std error sigma ($\AA$)' : unc_s}
    
    df = pd.DataFrame(params_to_save)
    df = df.replace(np.nan, 'NaN') 

    df.to_csv(os.path.join(outputPath, str(hkl) + '_peak_fit_results_' + sampleName + '.csv'), index=None)
        
    return

#%%
#PL-Fitting

def sum_of_Voigts(x, *params):
    
    if isinstance(x, float):
        x = np.array([x])

    params = np.array(params)
    n = (len(params)-2) // 4

    # divide parameters
    amps   = params[:n]
    mus    = params[n:2*n]
    sigmas = params[2*n:3*n]
    alphas = params[3*n:4*n]

    gaussians  = amps*np.exp(-(x[:, np.newaxis] - mus)**2 / sigmas)
    lorentian  = np.log(2) * (2/np.pi)**0.5 * (amps*sigmas / ((x[:, np.newaxis] - mus)**2 + sigmas*np.log(2)))
    background = params[-2]*x + params[-1]

    return np.dot(gaussians, 1-alphas) + np.dot(lorentian, alphas) + background

def background(x, y0, y1):
    return y0*x + y1

def fWHM_Voigt(x, center, maxValue, params):
    
    x1 = np.linspace(x[0], center, 5001)
    x2 = np.linspace(center, x[-1], 5001)
    
    y1 = sum_of_Voigts(x1, *params)
    y2 = sum_of_Voigts(x2, *params)
    
    root1 = np.interp(maxValue/2,y1,x1)
    root2 = np.interp(maxValue/2,y2[::-1],x2[::-1])
    
    return root2 - root1
 
def plFitting(plParams, df_yCut, df_xCutFit, df_fit, show_every, numGauss, peakLowerTH, inputDict, peakUpperTH, estPeakWidth, minPeakWidth, maxPeakWidth, name_d, name):
    
    estPositions = inputDict["PLFits_CenterGuesses"]
    
    frames = range(0, len(df_xCutFit))
    frames_to_plot = [i for i in frames if i % show_every == 0]

    yVals = np.copy(df_fit)
    popt = np.array([[np.nan, np.nan, np.nan, np.nan]*int(numGauss) + [np.nan, np.nan]] * np.shape(df_fit)[1])
    peakFWHM = np.array([[np.nan]*int(numGauss)] * np.shape(df_fit)[1])
    peakArea = np.array([[np.nan]*int(numGauss)] * np.shape(df_fit)[1])
    
    # The next block is to convert the estimated peak positions and ranges into indexes
    idxLowerTH = [0.0]*int(numGauss)
    idxUpperTH = [0.0]*int(numGauss)
    
    for i in range(0, int(numGauss)):
        idxLowerTH[i] = next(xStart for xStart, valStart in enumerate(df_yCut) if valStart > peakLowerTH[i])
        idxUpperTH[i] = next(xEnd for xEnd, valEnd in enumerate(df_yCut) if valEnd > peakUpperTH[i])
        
    firstSpectrum = True
    for i in range(0, np.shape(df_fit)[1]):
        
        # get y values
        yVals[:, i] = np.where(yVals[:, i] == float('inf'), 5, yVals[:, i])
        
        idx = np.argmax(yVals[0:idxUpperTH[0], i])
        yVals[idx, i] = yVals[idx - 1, i]
        
        # find peaks
        peaks = signal.find_peaks(yVals[:, i])[0]
        
        # array initialization
        estAmplitudes = [0.0]*int(numGauss)
        minAmplitudes = [0.0]*int(numGauss)
        maxAmplitudes = [0.0]*int(numGauss)
        estAlphas = [0.24]*int(numGauss)
        minAlphas = [0.0]*int(numGauss)
        maxAlphas = [1.0]*int(numGauss)
        minLinBkg = 0.0
        estLinBkg = 0.0
        maxLinBkg = 1000.0
        minConstBkg = 0.0
        estConstBkg = 0.0
        maxConstBkg = 1000.0

            
        # no peak, skip
        if len(peaks) == 0:
            print("Time:")
            print(df_xCutFit[i])
            print("No Peak Found")
            continue
        
        if firstSpectrum:
            firstSpectrum = False
            firstFitIdx = i
            
            # find initial parameters and bounds for peak amplitudes, having free peaks start more prominent than propagating ones
            for ii in range(0,int(numGauss)):
                if float(inputDict["PLFits_Propagate?"][ii]):
                    estAmplitudes[ii] = max(yVals[idxLowerTH[ii]:idxUpperTH[ii], i]) / 5
                    minAmplitudes[ii] = 0
                    maxAmplitudes[ii] = max(yVals[idxLowerTH[ii]:idxUpperTH[ii], i]) / 1.5
                else:
                    estAmplitudes[ii] = max(yVals[idxLowerTH[ii]:idxUpperTH[ii], i])
                    minAmplitudes[ii] = estAmplitudes[ii] / 10
                    maxAmplitudes[ii] = np.inf
            
            # collecting fit parameters
            estParams = estAmplitudes + estPositions + estPeakWidth + estAlphas + [estLinBkg, estConstBkg]
            lowerBounds = minAmplitudes + peakLowerTH + minPeakWidth + minAlphas + [minLinBkg, minConstBkg]
            upperBounds = maxAmplitudes + peakUpperTH + maxPeakWidth + maxAlphas + [maxLinBkg, maxConstBkg]
            
        else:
            # update initial parameters and bounds. Propagating peaks have their position and width linked to the first one 
            for ii in range(0,int(numGauss)):
                # estAlphas[ii] = popt[firstFitIdx, 3*int(numGauss)+ii]
                # minAlphas[ii] = estAlphas[ii] / 1.05
                # maxAlphas[ii] = estAlphas[ii] * 1.05
                
                if float(inputDict["PLFits_Propagate?"][ii]):
                    estAmplitudes[ii] = max(yVals[idxLowerTH[ii]:idxUpperTH[ii], i]) / 10
                    minAmplitudes[ii] = 0
                    maxAmplitudes[ii] = np.inf
                    estPositions[ii] = popt[firstFitIdx,int(numGauss)+ii]
                    peakLowerTH[ii] = estPositions[ii]
                    peakUpperTH[ii] = estPositions[ii] * 1.001
                    estPeakWidth[ii] = popt[firstFitIdx][2*int(numGauss)+ii]
                    minPeakWidth[ii] = estPeakWidth[ii] / 1.01
                    maxPeakWidth[ii] = estPeakWidth[ii] * 1.01

                else:
                    estAmplitudes[ii] = max(yVals[idxLowerTH[ii]:idxUpperTH[ii], i])
                    minAmplitudes[ii] = 0
                    maxAmplitudes[ii] = np.inf
                    
                   # # if previously converged, keep position from optimized (didn't improve fit much but makes it slower)
                   #  if not np.isnan(popt[i-1, int(numGauss)+ii]):
                   #      estPositions[ii] = popt[i-1,int(numGauss)+ii]
                   #      peakLowerTH[ii] = estPositions[ii] - 0.1
                   #      peakUpperTH[ii] = estPositions[ii] + 0.1
             
            # collecting fit parameters
            estParams   = estAmplitudes + estPositions + estPeakWidth + estAlphas + [estConstBkg, estLinBkg]
            lowerBounds = minAmplitudes + peakLowerTH + minPeakWidth + minAlphas + [0.0, 0.0]
            upperBounds = maxAmplitudes + peakUpperTH + maxPeakWidth + maxAlphas + [1000.0, 1000.0]

        # try fitting
        try:
            popt[i], pcov = curve_fit(sum_of_Voigts,
                                   df_yCut,
                                   yVals[:, i],
                                   p0     = estParams,
                                   bounds = (lowerBounds, upperBounds)
                                   )
            

        except Exception:
            print("Time:")
            print(df_xCutFit[i])
            traceback.print_exc()
            pass
        
        for ii in range(0,int(numGauss)):
            parameters = [popt[i,ii], popt[i,int(numGauss)+ii], popt[i,2*int(numGauss)+ii], popt[i,3*int(numGauss)+ii], 0, 0]
            peakFWHM[i,ii] = fWHM_Voigt(df_yCut, popt[i,int(numGauss)+ii], sum_of_Voigts(popt[i,int(numGauss)+ii], *parameters), parameters)
            peakArea[i,ii] = integrate.quad(lambda x: sum_of_Voigts(x, *parameters), -np.inf,np.inf)[0]
        
        # plotting fit results for pre-selected frames
        if i in frames_to_plot:
            
            plt.figure(figsize=(6, 5))
            plt.plot(df_yCut, yVals[:, i], 'o', label='data')
            plt.plot(df_yCut, sum_of_Voigts(df_yCut, *popt[i,:]), 'r-', label='fit')
            
            for ii in range(0, int(numGauss)):
                plt.plot(df_yCut, sum_of_Voigts(df_yCut, *[popt[i,ii], popt[i,int(numGauss)+ii], popt[i,2*int(numGauss)+ii], popt[i,3*int(numGauss)+ii], 0, 0]), '--', label='Peak ' + str(ii+1))
            plt.plot(df_yCut, background(df_yCut, *[popt[i,-2], popt[i,-1]]), 'k--', label='Background')              
            plt.legend()
            plt.xlabel('Energy (eV)')
            plt.ylabel('Intensity (a.u.)')
            plt.title('Time: ' + str(df_xCutFit[i]))
            plt.savefig(os.path.join(name + '/fits/', str(name_d) + '_PL-fit_' + str(int(df_xCutFit[i])) + '_s.png'), format = 'png')
            plt.show(block=False)
            plt.pause(1)
            
            if plParams['logplots']:
                plt.figure(figsize=(6, 5))
                plt.plot(df_yCut, np.log(yVals[:, i]), 'o', label='data')
                plt.plot(df_yCut, np.log(sum_of_Voigts(df_yCut, *popt[i,:])), 'r-', label='fit')
                plt.legend()
                plt.xlabel('Energy (eV)')
                plt.ylabel('Log-Intensity (a.u.)')
                plt.title('Time: ' + str(df_xCutFit[i]))
                plt.savefig(os.path.join(name + '/fits/', str(name_d) + '_PL-fit_Log_' + str(int(df_xCutFit[i])) + '_s.png'), format = 'png')
                plt.show(block=False)
                plt.pause(1)

    # Plotting the time-evolution of the peak-positions and intensities
    for i in range(0, int(numGauss)):
        fig, ax1 = plt.subplots(figsize=(6, 5))
        plot1, = ax1.plot(df_xCutFit, popt[:,int(numGauss)+i], label = 'Peak Position')
        ax2 = ax1.twinx()
        plot2, = ax2.plot(df_xCutFit, popt[:,i], 'g', label = 'Peak Intensity')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(r'PL Position (eV)')
        ax2.set_ylabel(r'PL Intensity (a.u.)')
        # Create your ticker object with M ticks
        yticks = ticker.MaxNLocator(5)
        ax1.yaxis.set_major_locator(yticks)
        fig.suptitle('Fit Results Peak ' + str(i+1) + ' ' + name_d, fontsize=14)
        fig.legend()
        
    # collecting the fit results in a dataframe
    dfPeaks = pd.DataFrame()
    dfPeaks['Fit-Time_' + name_d] = df_xCutFit
    for i in range(0,int(numGauss)):
        colPos = 'Peak' + str(i+1) + 'Pos_' + name_d
        colArea = 'Peak' + str(i+1) + 'Area_' + name_d
        colFWHM = 'Peak' + str(i+1) + 'FWHM_' + name_d
        colAlphas = 'Peak' + str(i+1) + 'Alpha_' + name_d
        data = np.array([peakArea[:,i], popt[:,int(numGauss)+i], peakFWHM[:,i], popt[:,3*int(numGauss)+i]])
        dfTemp = pd.DataFrame(
            data.T,
            columns=[colArea, colPos, colFWHM, colAlphas])
        dfPeaks = pd.concat([dfPeaks, dfTemp], axis=1)
    dfPeaks = dfPeaks.fillna('nan')

    # saving the data:
    dfPeaks.to_csv(str(name) + '/PL_FitResults.csv', index=False)
    
    return