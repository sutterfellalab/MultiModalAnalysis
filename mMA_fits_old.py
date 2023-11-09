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
                   'intercept' : 700    # 700
                   }

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
    peaks = signal.find_peaks(y, prominence=100)[0]
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
def constFit(x, y0):
    return y0 + x-x

def linFit(x, y0):
    return y0*x

def singularDecay(x, a1, x1, w1):
    return a1 * np.exp(-(x - x1) / w1)

def singularGauss(x, a1, x1, w1): 
    return a1 * np.exp(-(x - x1) ** 2 / w1)

def gauss(x, a1, x1, w1, y0):
    return y0*x + a1 * np.exp(-(x - x1) ** 2 / w1)

def decayGauss(x, a1, x1, w1, y0, a2, x2, w2):
    return y0 + a1 * np.exp(-(x - x1) ** 2 / w1) + a2 * np.exp(-(x - x2) / w2)

def doubleGauss(x, a1, x1, w1, y0, a2, x2, w2):
    return y0 + a1 * np.exp(-(x - x1) ** 2 / w1) + a2 * np.exp(-(x - x2) ** 2 / w2)

def doubleDecayGauss(x, a1, x1, w1, y0, a2, x2, w2, a3, x3, w3):
    return y0 + a1 * np.exp(-(x - x1) ** 2 / w1) + a2 * np.exp(-(x - x2) ** 2 / w2) + a3 * np.exp(-(x - x3) / w3)

def tripleGauss(x, a1, x1, w1, y0, a2, x2, w2, a3, x3, w3):
    return y0 + a1 * np.exp(-(x - x1) ** 2 / w1) + a2 * np.exp(-(x - x2) ** 2 / w2) + a3 * np.exp(-(x - x3) ** 2 / w3)

def tripleDecayGauss(x, a1, x1, w1, y0, a2, x2, w2, a3, x3, w3, a4, x4, w4):
    return y0 + a1 * np.exp(-(x - x1) ** 2 / w1) + a2 * np.exp(-(x - x2) ** 2 / w2) + a3 * np.exp(-(x - x3) ** 2 / w3) + a4 * np.exp(-(x - x4) / w4)

def fourGauss(x, a1, x1, w1, y0, a2, x2, w2, a3, x3, w3, a4, x4, w4):
    return y0 + a1 * np.exp(-(x - x1) ** 2 / w1) + a2 * np.exp(-(x - x2) ** 2 / w2) + a3 * np.exp(-(x - x3) ** 2 / w3) + a4 * np.exp(-(x - x4) ** 2 / w4)

def fourDecayGauss(x, a1, x1, w1, y0, a2, x2, w2, a3, x3, w3, a4, x4, w4, a5, x5, w5):
    return y0 + a1 * np.exp(-(x - x1) ** 2 / w1) + a2 * np.exp(-(x - x2) ** 2 / w2) + a3 * np.exp(-(x - x3) ** 2 / w3) + a4 * np.exp(-(x - x4) ** 2 / w4) + a5 * np.exp(-(x - x5) / w5)

def fiveGauss(x, a1, x1, w1, y0, a2, x2, w2, a3, x3, w3, a4, x4, w4, a5, x5, w5):
    return y0 + a1 * np.exp(-(x - x1) ** 2 / w1) + a2 * np.exp(-(x - x2) ** 2 / w2) + a3 * np.exp(-(x - x3) ** 2 / w3) + a4 * np.exp(-(x - x4) ** 2 / w4) + a5 * np.exp(-(x - x5) ** 2 / w5)

def sixGauss(x, a1, x1, w1, y0, a2, x2, w2, a3, x3, w3, a4, x4, w4, a5, x5, w5, a6, x6, w6):
    return y0 + a1 * np.exp(-(x - x1) ** 2 / w1) + a2 * np.exp(-(x - x2) ** 2 / w2) + a3 * np.exp(-(x - x3) ** 2 / w3) + a4 * np.exp(-(x - x4) ** 2 / w4) + a5 * np.exp(-(x - x5) ** 2 / w5) + a6 * np.exp(-(x - x6) ** 2 / w6)


def plFitting(plParams, df_yCut, df_xCutFit, df_fit, show_every, numGauss, peakLowerTH, peakStartPos, peakUpperTH, estPeakWidth, maxPeakWidth, maxBkg, name_d, name):
    
    frames = range(0, len(df_xCutFit))
    frames_to_plot = [i for i in frames if i % show_every == 0]
    
    peak1Maxs_Val = [0] * np.shape(df_fit)[1]
    peak1Maxs_Pos = [0] * np.shape(df_fit)[1]
    peak1Maxs_FWHM = [0] * np.shape(df_fit)[1]
    peak2Maxs_Val = [0] * np.shape(df_fit)[1]
    peak2Maxs_Pos = [0] * np.shape(df_fit)[1]
    peak2Maxs_FWHM = [0] * np.shape(df_fit)[1]
    peak3Maxs_Val = [0] * np.shape(df_fit)[1]
    peak3Maxs_Pos = [0] * np.shape(df_fit)[1]
    peak3Maxs_FWHM = [0] * np.shape(df_fit)[1]
    peak4Maxs_Val = [0] * np.shape(df_fit)[1]
    peak4Maxs_Pos = [0] * np.shape(df_fit)[1]
    peak4Maxs_FWHM = [0] * np.shape(df_fit)[1]
    peak5Maxs_Val = [0] * np.shape(df_fit)[1]
    peak5Maxs_Pos = [0] * np.shape(df_fit)[1]
    peak5Maxs_FWHM = [0] * np.shape(df_fit)[1]
    peak6Maxs_Val = [0] * np.shape(df_fit)[1]
    peak6Maxs_Pos = [0] * np.shape(df_fit)[1]
    peak6Maxs_FWHM = [0] * np.shape(df_fit)[1]
    
    yVals = np.copy(df_fit)
    popt = [[np.nan, np.nan, np.nan, np.nan]] * np.shape(df_fit)[1]
    
    fitDict = {'numPeaks': numGauss, 'fitFunction': (), 'fitEstimates': (), 'fitMinima': (), 'fitMaxima': ()}

    # The next block is to convert the estimated peak positions and ranges into indexes
    idxLowerTH = np.array([0, 0, 0, 0, 0, 0])
    idxUpperTH = np.array([0, 0, 0, 0, 0, 0])

    for i in range(0, int(numGauss)):
        idxLowerTH[i] = next(xStart for xStart, valStart in enumerate(df_yCut) if valStart > peakLowerTH[i])
        idxUpperTH[i] = next(xEnd for xEnd, valEnd in enumerate(df_yCut) if valEnd > peakUpperTH[i])

    for i in range(0, np.shape(df_fit)[1]):
        yVals[:, i] = np.where(yVals[:, i] == float('inf'), 5, yVals[:, i])
        idx = np.argmax(yVals[0:idxUpperTH[0], i])
        yVals[idx, i] = yVals[idx - 1, i]
        
        peaks = signal.find_peaks(yVals[:, i], prominence=10)[0]
        if len(peaks) > 0:
            
            try:
                if numGauss == 0:
                    idx = np.argmax(yVals[0:idxUpperTH[0], i])
                    yVals[idx, i] = yVals[idx-1, i]
                    idx = np.argmax(yVals[0:idxUpperTH[0], i])
                    peak2Maxs_Pos[i] = df_yCut[idx]
                elif numGauss == 1:
                    popt[i], pcov = curve_fit(gauss, df_yCut, yVals[:, i], p0=[max(yVals[idxLowerTH[0]:idxUpperTH[0], i]), peakStartPos[0], estPeakWidth, 
                                                                               0.1], 
                                              maxfev=5000, bounds=((0, 0, 0, -np.inf),
                                                                   (np.inf, np.inf, maxPeakWidth, maxBkg)))
                elif numGauss == 1.5:
                    popt[i], pcov = curve_fit(decayGauss, df_yCut, yVals[:, i], p0=[max(yVals[:, i]), peakStartPos[0], estPeakWidth, 
                                                                                    0.1, 
                                                                                    10, peakStartPos[0]/2, estPeakWidth], 
                                              maxfev=5000, bounds=((0, 0, 0, 0, 0, 0, 0), 
                                                                   (np.inf, np.inf, maxPeakWidth, maxBkg, np.inf, np.inf, maxPeakWidth)))
                elif numGauss == 2:
                    popt[i], pcov = curve_fit(doubleGauss, df_yCut, yVals[:, i], p0=[max(yVals[idxLowerTH[0]:idxUpperTH[0], i]), peakStartPos[0], estPeakWidth,
                                                                                     0.1, 
                                                                                     max(yVals[idxLowerTH[1]:idxUpperTH[1], i]), peakStartPos[1], estPeakWidth],
                                              maxfev=5000, bounds=((max(yVals[idxLowerTH[0]:idxUpperTH[0], i]) / 1.2, peakLowerTH[0], 0, 
                                                                    0, 
                                                                    max(yVals[idxLowerTH[1]:idxUpperTH[1], i]) / 1.2, peakLowerTH[1], 0), 
                                                                   (max(yVals[idxLowerTH[0]:idxUpperTH[0], i]) * 1.2, peakUpperTH[0], maxPeakWidth, 
                                                                    maxBkg, 
                                                                    max(yVals[idxLowerTH[1]:idxUpperTH[1], i]) * 1.2, peakUpperTH[1], maxPeakWidth)))
                    peak2Maxs_Val[i] = popt[i][4]
                    peak2Maxs_Pos[i] = popt[i][5]
                    peak2Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][6])
                elif numGauss == 2.5:
                    popt[i], pcov = curve_fit(doubleDecayGauss, df_yCut, yVals[:, i], p0=[max(yVals[idxLowerTH[0]:idxUpperTH[0], i]), peakStartPos[0], estPeakWidth,
                                                                                     yVals[-1, i] - 1, 
                                                                                     max(yVals[idxLowerTH[1]:idxUpperTH[1], i]), peakStartPos[1], estPeakWidth,
                                                                                     yVals[0, i], df_yCut[0], 0.01],
                                              maxfev=5000, bounds=((max(yVals[idxLowerTH[0]:idxUpperTH[0], i]) / 1.2, peakLowerTH[0], 0, 
                                                                    - np.inf, 
                                                                    max(yVals[idxLowerTH[1]:idxUpperTH[1], i]) / 1.2, peakLowerTH[1], 0,
                                                                    0, 0, 0), 
                                                                   (max(yVals[idxLowerTH[0]:idxUpperTH[0], i]) * 1.2, peakUpperTH[0], maxPeakWidth, 
                                                                    yVals[-1, i], 
                                                                    max(yVals[idxLowerTH[1]:idxUpperTH[1], i]) * 1.2, peakUpperTH[1], maxPeakWidth,
                                                                    np.inf, np.inf, np.inf)))
                    peak2Maxs_Val[i] = popt[i][4]
                    peak2Maxs_Pos[i] = popt[i][5]
                    peak2Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][6])    
                elif numGauss == 3:
                    popt[i], pcov = curve_fit(tripleGauss, df_yCut, yVals[:, i], p0=[max(yVals[idxLowerTH[0]:idxUpperTH[0], i]), peakStartPos[0], estPeakWidth, 
                                                                                     0.1, 
                                                                                     max(yVals[idxLowerTH[1]:idxUpperTH[1], i]), peakStartPos[1], estPeakWidth, 
                                                                                     max(yVals[idxLowerTH[2]:idxUpperTH[2], i]), peakStartPos[2], estPeakWidth],
                                              maxfev=5000, bounds=((max(yVals[idxLowerTH[0]:idxUpperTH[0], i]) / 1.2, peakLowerTH[0], 0, 
                                                                    0, 
                                                                    max(yVals[idxLowerTH[1]:idxUpperTH[1], i]) / 1.2, peakLowerTH[1], 0, 
                                                                    max(yVals[idxLowerTH[2]:idxUpperTH[2], i]) / 1.2, peakLowerTH[2], 0), 
                                                                   (max(yVals[idxLowerTH[0]:idxUpperTH[0], i]) * 1.2, peakUpperTH[0], maxPeakWidth, 
                                                                    maxBkg, 
                                                                    max(yVals[idxLowerTH[1]:idxUpperTH[1], i]) * 1.2, peakUpperTH[1], maxPeakWidth, 
                                                                    max(yVals[idxLowerTH[2]:idxUpperTH[2], i]) * 1.2, peakUpperTH[2], maxPeakWidth)))
                    peak2Maxs_Val[i] = popt[i][4]
                    peak2Maxs_Pos[i] = popt[i][5]
                    peak2Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][6])
                    peak3Maxs_Val[i] = popt[i][7]
                    peak3Maxs_Pos[i] = popt[i][8]
                    peak3Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][9])
                elif numGauss == 3.5:
                    popt[i], pcov = curve_fit(tripleDecayGauss, df_yCut, yVals[:, i], p0=[max(yVals[idxLowerTH[0]:idxUpperTH[0], i]), peakStartPos[0], estPeakWidth, 
                                                                                     yVals[-1, i] - 1, 
                                                                                     max(yVals[idxLowerTH[1]:idxUpperTH[1], i]), peakStartPos[1], estPeakWidth, 
                                                                                     max(yVals[idxLowerTH[2]:idxUpperTH[2], i]), peakStartPos[2], estPeakWidth,
                                                                                     yVals[0, i], df_yCut[0], 0.01],
                                              maxfev=5000, bounds=((max(yVals[idxLowerTH[0]:idxUpperTH[0], i]) / 1.2, peakLowerTH[0], 0, 
                                                                    - np.inf, 
                                                                    max(yVals[idxLowerTH[1]:idxUpperTH[1], i]) / 1.2, peakLowerTH[1], 0, 
                                                                    max(yVals[idxLowerTH[2]:idxUpperTH[2], i]) / 1.2, peakLowerTH[2], 0,
                                                                    0, 0, 0), 
                                                                   (max(yVals[idxLowerTH[0]:idxUpperTH[0], i]) * 1.2, peakUpperTH[0], maxPeakWidth, 
                                                                    yVals[-1, i], 
                                                                    max(yVals[idxLowerTH[1]:idxUpperTH[1], i]) * 1.2, peakUpperTH[1], maxPeakWidth, 
                                                                    max(yVals[idxLowerTH[2]:idxUpperTH[2], i]) * 1.2, peakUpperTH[2], maxPeakWidth,
                                                                    np.inf, np.inf, np.inf)))
                    peak2Maxs_Val[i] = popt[i][4]
                    peak2Maxs_Pos[i] = popt[i][5]
                    peak2Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][6])
                    peak3Maxs_Val[i] = popt[i][7]
                    peak3Maxs_Pos[i] = popt[i][8]
                    peak3Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][9])
                elif numGauss == 4:
                    popt[i], pcov = curve_fit(fourGauss, df_yCut, yVals[:, i], p0=[max(yVals[idxLowerTH[0]:idxUpperTH[0], i]), peakStartPos[0], estPeakWidth, 
                                                                                   0.1, 
                                                                                   max(yVals[idxLowerTH[1]:idxUpperTH[1], i]), peakStartPos[1], estPeakWidth, 
                                                                                   max(yVals[idxLowerTH[2]:idxUpperTH[2], i]), peakStartPos[2], estPeakWidth, 
                                                                                   max(yVals[idxLowerTH[3]:idxUpperTH[3], i]), peakStartPos[3], estPeakWidth], 
                                              maxfev=5000, bounds=((max(yVals[idxLowerTH[0]:idxUpperTH[0], i])/1.2, peakLowerTH[0], 0, 
                                                                    0, 
                                                                    0, peakLowerTH[1], 0, 
                                                                    0, peakLowerTH[2], 0, 
                                                                    0, peakLowerTH[3], 0), 
                                                                   (max(yVals[idxLowerTH[0]:idxUpperTH[0], i])*1.2, peakUpperTH[0], maxPeakWidth, 
                                                                    maxBkg, 
                                                                    np.inf, peakUpperTH[1], maxPeakWidth, 
                                                                    np.inf, peakUpperTH[2], maxPeakWidth, 
                                                                    np.inf, peakUpperTH[3], maxPeakWidth)))
                    peak2Maxs_Val[i] = popt[i][4]
                    peak2Maxs_Pos[i] = popt[i][5]
                    peak2Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][6])
                    peak3Maxs_Val[i] = popt[i][7]
                    peak3Maxs_Pos[i] = popt[i][8]
                    peak3Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][9])
                    peak4Maxs_Val[i] = popt[i][10]
                    peak4Maxs_Pos[i] = popt[i][11]
                    peak4Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][12])
                elif numGauss == 4.5:
                    popt[i], pcov = curve_fit(fourDecayGauss, df_yCut, yVals[:, i], p0=[max(yVals[idxLowerTH[0]:idxUpperTH[0], i]), peakStartPos[0], estPeakWidth, 
                                                                                   yVals[-1, i] - 1, 
                                                                                   max(yVals[idxLowerTH[1]:idxUpperTH[1], i]), peakStartPos[1], estPeakWidth, 
                                                                                   max(yVals[idxLowerTH[2]:idxUpperTH[2], i]), peakStartPos[2], estPeakWidth, 
                                                                                   max(yVals[idxLowerTH[3]:idxUpperTH[3], i]), peakStartPos[3], estPeakWidth,
                                                                                   yVals[0, i], df_yCut[0], 0.01], 
                                              maxfev=5000, bounds=((max(yVals[idxLowerTH[0]:idxUpperTH[0], i])/1.2, peakLowerTH[0], 0, 
                                                                    - np.inf, 
                                                                    max(yVals[idxLowerTH[1]:idxUpperTH[1], i]) / 1.2, peakLowerTH[1], 0, 
                                                                    max(yVals[idxLowerTH[2]:idxUpperTH[2], i]) / 1.2, peakLowerTH[2], 0, 
                                                                    max(yVals[idxLowerTH[3]:idxUpperTH[3], i]) / 1.2, peakLowerTH[3], 0, 
                                                                    0, 0, 0),
                                                                   (max(yVals[idxLowerTH[0]:idxUpperTH[0], i])*1.2, peakUpperTH[0], maxPeakWidth, 
                                                                    yVals[-1, i], 
                                                                    max(yVals[idxLowerTH[1]:idxUpperTH[1], i]) * 1.2, peakUpperTH[1], maxPeakWidth, 
                                                                    max(yVals[idxLowerTH[2]:idxUpperTH[2], i]) * 1.2, peakUpperTH[2], maxPeakWidth,
                                                                    max(yVals[idxLowerTH[3]:idxUpperTH[3], i]) * 1.2, peakUpperTH[3], maxPeakWidth,
                                                                    np.inf, np.inf, np.inf)))
                    peak2Maxs_Val[i] = popt[i][4]
                    peak2Maxs_Pos[i] = popt[i][5]
                    peak2Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][6])
                    peak3Maxs_Val[i] = popt[i][7]
                    peak3Maxs_Pos[i] = popt[i][8]
                    peak3Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][9])
                    peak4Maxs_Val[i] = popt[i][10]
                    peak4Maxs_Pos[i] = popt[i][11]
                    peak4Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][12])
                elif numGauss == 5:
                    popt[i], pcov = curve_fit(fiveGauss, df_yCut, yVals[:, i], p0=[max(yVals[idxLowerTH[0]:idxUpperTH[0], i]), peakStartPos[0], estPeakWidth, 
                                                                                   0.1, 
                                                                                   max(yVals[idxLowerTH[1]:idxUpperTH[1], i]), peakStartPos[1], estPeakWidth, 
                                                                                   max(yVals[idxLowerTH[2]:idxUpperTH[2], i]), peakStartPos[2], estPeakWidth, 
                                                                                   max(yVals[idxLowerTH[3]:idxUpperTH[3], i]), peakStartPos[3], estPeakWidth, 
                                                                                   max(yVals[idxLowerTH[4]:idxUpperTH[4], i]), peakStartPos[4], estPeakWidth], 
                                              maxfev=2000, bounds=((max(yVals[idxLowerTH[0]:idxUpperTH[0], i])/1.75, peakLowerTH[0], 0, 
                                                                    0, 
                                                                    0, peakLowerTH[1], 0, 
                                                                    0, peakLowerTH[2], 0, 
                                                                    0, peakLowerTH[3], 0, 
                                                                    0, peakLowerTH[4], 0), 
                                                                   (max(yVals[idxLowerTH[0]:idxUpperTH[0], i])*1.75, peakUpperTH[0], maxPeakWidth, 
                                                                    maxBkg, 
                                                                    np.inf, peakUpperTH[1], maxPeakWidth, 
                                                                    np.inf, peakUpperTH[2], maxPeakWidth, 
                                                                    np.inf, peakUpperTH[3], maxPeakWidth, 
                                                                    np.inf, peakUpperTH[4], maxPeakWidth)))
                    peak2Maxs_Val[i] = popt[i][4]
                    peak2Maxs_Pos[i] = popt[i][5]
                    peak2Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][6])
                    peak3Maxs_Val[i] = popt[i][7]
                    peak3Maxs_Pos[i] = popt[i][8]
                    peak3Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][9])
                    peak4Maxs_Val[i] = popt[i][10]
                    peak4Maxs_Pos[i] = popt[i][11]
                    peak4Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][12])
                    peak5Maxs_Val[i] = popt[i][13]
                    peak5Maxs_Pos[i] = popt[i][14]
                    peak5Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][15])
                elif numGauss == 6:
                    popt[i], pcov = curve_fit(sixGauss, df_yCut, yVals[:, i], p0=[max(yVals[idxLowerTH[0]:idxUpperTH[0], i]), peakStartPos[0], estPeakWidth, 
                                                                                  0.1, 
                                                                                  max(yVals[idxLowerTH[1]:idxUpperTH[1], i]), peakStartPos[1], estPeakWidth, 
                                                                                  max(yVals[idxLowerTH[2]:idxUpperTH[2], i]), peakStartPos[2], estPeakWidth, 
                                                                                  max(yVals[idxLowerTH[3]:idxUpperTH[3], i]), peakStartPos[3], estPeakWidth, 
                                                                                  max(yVals[idxLowerTH[4]:idxUpperTH[4], i]), peakStartPos[4], estPeakWidth, 
                                                                                  max(yVals[idxLowerTH[5]:idxUpperTH[5], i]), peakStartPos[5], estPeakWidth], 
                                              maxfev=5000, bounds=((max(yVals[idxLowerTH[0]:idxUpperTH[0], i])/1.75, peakLowerTH[0], 0, 
                                                                    0, 
                                                                    0, peakLowerTH[1], 0, 
                                                                    0, peakLowerTH[2], 0, 
                                                                    0, peakLowerTH[3], 0, 
                                                                    0, peakLowerTH[4], 0, 
                                                                    0, peakLowerTH[5], 0), 
                                                                   (max(yVals[idxLowerTH[0]:idxUpperTH[0], i])*1.75, peakUpperTH[0], maxPeakWidth, 
                                                                    maxBkg, 
                                                                    np.inf, peakUpperTH[1], maxPeakWidth, 
                                                                    np.inf, peakUpperTH[2], maxPeakWidth, 
                                                                    np.inf, peakUpperTH[3], maxPeakWidth, 
                                                                    np.inf, peakUpperTH[4], maxPeakWidth, 
                                                                    np.inf, peakUpperTH[5], maxPeakWidth)))
                    peak2Maxs_Val[i] = popt[i][4]
                    peak2Maxs_Pos[i] = popt[i][5]
                    peak2Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][6])
                    peak3Maxs_Val[i] = popt[i][7]
                    peak3Maxs_Pos[i] = popt[i][8]
                    peak3Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][9])
                    peak4Maxs_Val[i] = popt[i][10]
                    peak4Maxs_Pos[i] = popt[i][11]
                    peak4Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][12])
                    peak5Maxs_Val[i] = popt[i][13]
                    peak5Maxs_Pos[i] = popt[i][14]
                    peak5Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][15])
                    peak6Maxs_Val[i] = popt[i][16]
                    peak6Maxs_Pos[i] = popt[i][17]
                    peak6Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][18])
    
    
                peak1Maxs_Val[i] = popt[i][0]
                peak1Maxs_Pos[i] = popt[i][1]
                peak1Maxs_FWHM[i] = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5 * popt[i][2])

                if i in frames_to_plot:

                    plt.figure(figsize=(6, 5))
                    plt.plot(df_yCut, yVals[:, i], 'o', label='data')
                    
                    if numGauss == 1:
                        plt.plot(df_yCut, gauss(df_yCut, *popt[i]), 'r-', label='fit')
                    elif numGauss == 1.5:
                        plt.plot(df_yCut, decayGauss(df_yCut, *popt[i]), 'r-', label='fit')
                    elif numGauss == 2:
                        plt.plot(df_yCut, doubleGauss(df_yCut, *popt[i]), 'r-', label='fit')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][0:3]), 'g--', label='Peak 1')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][4:7]), 'c--', label='Peak 2')
                    elif numGauss == 2.5:
                        plt.plot(df_yCut, doubleDecayGauss(df_yCut, *popt[i]), 'r-', label='fit')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][0:3]), 'g--', label='Peak 1')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][4:7]), 'c--', label='Peak 2')
                        plt.plot(df_yCut, singularDecay(df_yCut, *popt[i][7:10]) + popt[i][3], 'k--', label='Bkg')                      
                    elif numGauss == 3:
                        plt.plot(df_yCut, tripleGauss(df_yCut, *popt[i]), 'r-', label='fit')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][0:3]), 'g--', label='Peak 1')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][4:7]), 'c--', label='Peak 2')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][7:10]), 'm--', label='Peak 3')
                    elif numGauss == 3.5:
                        plt.plot(df_yCut, tripleDecayGauss(df_yCut, *popt[i]), 'r-', label='fit')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][0:3]), 'g--', label='Peak 1')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][4:7]), 'c--', label='Peak 2')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][7:10]), 'm--', label='Peak 3')
                        plt.plot(df_yCut, singularDecay(df_yCut, *popt[i][10:13]) + popt[i][3], 'k--', label='Bkg')
                    elif numGauss == 4:
                        plt.plot(df_yCut, fourGauss(df_yCut, *popt[i]), 'r-', label='fit')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][0:3]), 'g--', label='Peak 1')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][4:7]), 'c--', label='Peak 2')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][7:10]), 'm--', label='Peak 3')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][10:13]), 'y--', label='Peak 4')
                        plt.plot(df_yCut, constFit(df_yCut, popt[i][3]), 'k--', label='Bkg')
                    elif numGauss == 4.5:
                        plt.plot(df_yCut, fourDecayGauss(df_yCut, *popt[i]), 'r-', label='fit')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][0:3]), 'g--', label='Peak 1')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][4:7]), 'c--', label='Peak 2')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][7:10]), 'm--', label='Peak 3')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][10:13]), 'y--', label='Peak 4')
                        plt.plot(df_yCut, singularDecay(df_yCut, *popt[i][13:16]) + popt[i][3], 'k--', label='Bkg')
                    elif numGauss == 5:
                        plt.plot(df_yCut, fiveGauss(df_yCut, *popt[i]), 'r-', label='fit')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][0:3]), 'g--', label='Peak 1')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][4:7]), 'c--', label='Peak 2')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][7:10]), 'm--', label='Peak 3')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][10:13]), 'y--', label='Peak 4')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][13:16]), '--', color='orange', label='Peak 5')
                        plt.plot(df_yCut, constFit(df_yCut, popt[i][3]), 'k--', label='Bkg')
                    elif numGauss == 6:
                        plt.plot(df_yCut, sixGauss(df_yCut, *popt[i]), 'r-', label='fit')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][0:3]), 'g--', label='Peak 1')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][4:7]), 'c--', label='Peak 2')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][7:10]), 'm--', label='Peak 3')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][10:13]), 'y--', label='Peak 4')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][13:16]), '--', color='orange', label='Peak 5')
                        plt.plot(df_yCut, singularGauss(df_yCut, *popt[i][16:19]), '--', color='pink', label='Peak 6')
                        plt.plot(df_yCut, constFit(df_yCut, popt[i][3]), 'k--', label='Bkg')
                
                    plt.legend()
                    plt.xlabel('Energy (eV)')
                    plt.ylabel('Intensity (a.u.)')
                    plt.title('Time: ' + str(df_xCutFit[i]))
                    plt.savefig(os.path.join(name + '/fits/', str(name_d) + '_PL-fit_' + str(int(df_xCutFit[i])) + '_s.png'), format = 'png')
                    plt.show(block=False)
                    plt.pause(1)
                    
                    plt.figure(figsize=(6, 5))
                    plt.plot(df_yCut, np.log(yVals[:, i]), 'o', label='data')
                    if plParams['logplots']:
                        if numGauss == 1:
                            plt.plot(df_yCut, np.log(gauss(df_yCut, *popt[i])), 'r-', label='fit')
                        elif numGauss == 1.5:
                            plt.plot(df_yCut, np.log(decayGauss(df_yCut, *popt[i])), 'r-', label='fit')
                        elif numGauss == 2:
                            plt.plot(df_yCut, np.log(doubleGauss(df_yCut, *popt[i])), 'r-', label='fit')
                        elif numGauss == 2.5:
                            plt.plot(df_yCut, np.log(doubleDecayGauss(df_yCut, *popt[i])), 'r-', label='fit')
                        elif numGauss == 3:
                            plt.plot(df_yCut, np.log(tripleGauss(df_yCut, *popt[i])), 'r-', label='fit')
                        elif numGauss == 3.5:
                            plt.plot(df_yCut, np.log(tripleDecayGauss(df_yCut, *popt[i])), 'r-', label='fit')
                        elif numGauss == 4:
                            plt.plot(df_yCut, np.log(fourGauss(df_yCut, *popt[i])), 'r-', label='fit')
                        elif numGauss == 4.5:
                            plt.plot(df_yCut, np.log(fourDecayGauss(df_yCut, *popt[i])), 'r-', label='fit')
                        elif numGauss == 5:
                            plt.plot(df_yCut, np.log(fiveGauss(df_yCut, *popt[i])), 'r-', label='fit')
                        elif numGauss == 6:
                            plt.plot(df_yCut, np.log(sixGauss(df_yCut, *popt[i])), 'r-', label='fit')
                    
                        plt.legend()
                        plt.xlabel('Energy (eV)')
                        plt.ylabel('Log-Intensity (a.u.)')
                        plt.title('Time: ' + str(df_xCutFit[i]))
                        plt.savefig(os.path.join(name + '/fits/', str(name_d) + '_PL-fit_Log_' + str(int(df_xCutFit[i])) + '_s.png'), format = 'png')
                        plt.show(block=False)
                        plt.pause(1)

            except Exception:
                print("Time:")
                print(df_xCutFit[i])
                traceback.print_exc()
                pass
            
        elif len(peaks) == 0:
            popt[i] = [None] * int(numGauss) * 4

    # Plotting the time-evolution of the peak-maximum
    fig, ax1 = plt.subplots(figsize=(6, 5))
    plot1, = ax1.plot(df_xCutFit, peak1Maxs_Pos, label = 'Peak Position')
    ax2 = ax1.twinx()
    plot2, = ax2.plot(df_xCutFit, peak1Maxs_Val, 'g', label = 'Peak Intensity')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(r'PL Position (eV)')
    ax2.set_ylabel(r'PL Intensity (a.u.)')
    # Create your ticker object with M ticks
    yticks = ticker.MaxNLocator(5)
    ax1.yaxis.set_major_locator(yticks)
    fig.suptitle('Fit Results Peak 1 ' + name_d, fontsize=14)
    fig.legend()
     
    if numGauss == 1:
        dfPeaks = pd.DataFrame(list(zip(df_xCutFit, peak1Maxs_Pos, peak1Maxs_Val, peak1Maxs_FWHM)), columns=(['Fit-Time_' + name_d, 'Peak1Pos_' + name_d, 'Peak1Height_' + name_d, 'Peak1FWHM_' + name_d]))
    elif numGauss == 1.5:
        dfPeaks = pd.DataFrame(list(zip(df_xCutFit, peak1Maxs_Pos, peak1Maxs_Val, peak1Maxs_FWHM)), columns=(['Fit-Time_' + name_d, 'Peak1Pos_' + name_d, 'Peak1Height_' + name_d, 'Peak1FWHM_' + name_d]))
    elif numGauss == 2:
        dfPeaks = pd.DataFrame(list(zip(df_xCutFit, peak1Maxs_Pos, peak1Maxs_Val, peak1Maxs_FWHM, peak2Maxs_Pos, peak2Maxs_Val, peak2Maxs_FWHM)), columns=(['Fit-Time_' + name_d, 'Peak1Pos_' + name_d, 'Peak1Height_' + name_d, 'Peak1FWHM_' + name_d,  'Peak2Pos_' + name_d, 'Peak2Height_' + name_d, 'Peak2FWHM_' + name_d]))
    elif numGauss == 2.5:
        dfPeaks = pd.DataFrame(list(zip(df_xCutFit, peak1Maxs_Pos, peak1Maxs_Val, peak1Maxs_FWHM, peak2Maxs_Pos, peak2Maxs_Val, peak2Maxs_FWHM)), columns=(['Fit-Time_' + name_d, 'Peak1Pos_' + name_d, 'Peak1Height_' + name_d, 'Peak1FWHM_' + name_d,  'Peak2Pos_' + name_d, 'Peak2Height_' + name_d, 'Peak2FWHM_' + name_d]))
    elif numGauss == 3:
        dfPeaks = pd.DataFrame(list(zip(df_xCutFit, peak1Maxs_Pos, peak1Maxs_Val, peak1Maxs_FWHM, peak2Maxs_Pos, peak2Maxs_Val, peak2Maxs_FWHM, peak3Maxs_Pos, peak3Maxs_Val, peak3Maxs_FWHM)), columns=(['Fit-Time_' + name_d, 'Peak1Pos_' + name_d, 'Peak1Height_' + name_d, 'Peak1FWHM_' + name_d,  'Peak2Pos_' + name_d, 'Peak2Height_' + name_d, 'Peak2FWHM_' + name_d,  'Peak3Pos_' + name_d, 'Peak3Height_' + name_d, 'Peak3FWHM_' + name_d]))
    elif numGauss == 3.5:
        dfPeaks = pd.DataFrame(list(zip(df_xCutFit, peak1Maxs_Pos, peak1Maxs_Val, peak1Maxs_FWHM, peak2Maxs_Pos, peak2Maxs_Val, peak2Maxs_FWHM, peak3Maxs_Pos, peak3Maxs_Val, peak3Maxs_FWHM)), columns=(['Fit-Time_' + name_d, 'Peak1Pos_' + name_d, 'Peak1Height_' + name_d, 'Peak1FWHM_' + name_d,  'Peak2Pos_' + name_d, 'Peak2Height_' + name_d, 'Peak2FWHM_' + name_d,  'Peak3Pos_' + name_d, 'Peak3Height_' + name_d, 'Peak3FWHM_' + name_d]))
    elif numGauss == 4:
        dfPeaks = pd.DataFrame(list(zip(df_xCutFit, peak1Maxs_Pos, peak1Maxs_Val, peak1Maxs_FWHM, peak2Maxs_Pos, peak2Maxs_Val, peak2Maxs_FWHM, peak3Maxs_Pos, peak3Maxs_Val, peak3Maxs_FWHM, peak4Maxs_Pos, peak4Maxs_Val, peak4Maxs_FWHM)), columns=(['Fit-Time_' + name_d, 'Peak1Pos_' + name_d, 'Peak1Height_' + name_d, 'Peak1FWHM_' + name_d,  'Peak2Pos_' + name_d, 'Peak2Height_' + name_d, 'Peak2FWHM_' + name_d,  'Peak3Pos_' + name_d, 'Peak3Height_' + name_d, 'Peak3FWHM_' + name_d,  'Peak4Pos_' + name_d, 'Peak4Height_' + name_d, 'Peak4FWHM_' + name_d]))
    elif numGauss == 4.5:
        dfPeaks = pd.DataFrame(list(zip(df_xCutFit, peak1Maxs_Pos, peak1Maxs_Val, peak1Maxs_FWHM, peak2Maxs_Pos, peak2Maxs_Val, peak2Maxs_FWHM, peak3Maxs_Pos, peak3Maxs_Val, peak3Maxs_FWHM, peak4Maxs_Pos, peak4Maxs_Val, peak4Maxs_FWHM)), columns=(['Fit-Time_' + name_d, 'Peak1Pos_' + name_d, 'Peak1Height_' + name_d, 'Peak1FWHM_' + name_d,  'Peak2Pos_' + name_d, 'Peak2Height_' + name_d, 'Peak2FWHM_' + name_d,  'Peak3Pos_' + name_d, 'Peak3Height_' + name_d, 'Peak3FWHM_' + name_d,  'Peak4Pos_' + name_d, 'Peak4Height_' + name_d, 'Peak4FWHM_' + name_d]))
    elif numGauss == 5:
        dfPeaks = pd.DataFrame(list(zip(df_xCutFit, peak1Maxs_Pos, peak1Maxs_Val, peak1Maxs_FWHM, peak2Maxs_Pos, peak2Maxs_Val, peak2Maxs_FWHM, peak3Maxs_Pos, peak3Maxs_Val, peak3Maxs_FWHM, peak4Maxs_Pos, peak4Maxs_Val, peak4Maxs_FWHM, peak5Maxs_Pos, peak5Maxs_Val, peak5Maxs_FWHM)), columns=(['Fit-Time_' + name_d, 'Peak1Pos_' + name_d, 'Peak1Height_' + name_d, 'Peak1FWHM_' + name_d,  'Peak2Pos_' + name_d, 'Peak2Height_' + name_d, 'Peak2FWHM_' + name_d,  'Peak3Pos_' + name_d, 'Peak3Height_' + name_d, 'Peak3FWHM_' + name_d,  'Peak4Pos_' + name_d, 'Peak4Height_' + name_d, 'Peak4FWHM_' + name_d,  'Peak5Pos_' + name_d, 'Peak5Height_' + name_d, 'Peak5FWHM_' + name_d]))
    elif numGauss == 6:
        dfPeaks = pd.DataFrame(list(zip(df_xCutFit, peak1Maxs_Pos, peak1Maxs_Val, peak1Maxs_FWHM, peak2Maxs_Pos, peak2Maxs_Val, peak2Maxs_FWHM, peak3Maxs_Pos, peak3Maxs_Val, peak3Maxs_FWHM, peak4Maxs_Pos, peak4Maxs_Val, peak4Maxs_FWHM, peak5Maxs_Pos, peak5Maxs_Val, peak5Maxs_FWHM, peak6Maxs_Pos, peak6Maxs_Val, peak6Maxs_FWHM)), columns=(['Fit-Time_' + name_d, 'Peak1Pos_' + name_d, 'Peak1Height_' + name_d, 'Peak1FWHM_' + name_d,  'Peak2Pos_' + name_d, 'Peak2Height_' + name_d, 'Peak2FWHM_' + name_d,  'Peak3Pos_' + name_d, 'Peak3Height_' + name_d, 'Peak3FWHM_' + name_d,  'Peak4Pos_' + name_d, 'Peak4Height_' + name_d, 'Peak4FWHM_' + name_d,  'Peak5Pos_' + name_d, 'Peak5Height_' + name_d, 'Peak5FWHM_' + name_d,  'Peak6Pos_' + name_d, 'Peak6Height_' + name_d, 'Peak6FWHM_' + name_d]))


    # saving the data:
    dfPeaks = dfPeaks.fillna(value=np.nan)
    dfPeaks.to_csv(str(name) + '/PL_FitResults.csv', index=False)
    
    return