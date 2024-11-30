# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:27:35 2022

@author: Tim Kodalle
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from bokeh.palettes import Viridis256, Greys256
from bokeh.plotting import figure, output_file, save
from bokeh.models import LinearColorMapper, ColorBar, NumericInput, LinearAxis, Range1d, HoverTool, CheckboxGroup, CustomJS
from bokeh.models.layouts import TabPanel, Tabs
from bokeh.layouts import layout

def plotGIWAXS(sample_name, save_path, q, frame_time, intensity):

    '''
    Parameters
    ----------
    sample_name : str,
        name of the sample. Default is the name under which scan is saved.
    save_path : path object
        where the output is saved.


    Returns
    -------
    Contour plot

    '''

    # create an empty figure with the following dimensions
    fig = plt.figure(figsize=(7, 5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])

    # add the contour plot and a colorbar
    cp = plt.contourf(frame_time, q, intensity.T)
    plt.colorbar(cp, location='left')

    # define axis names, ticks, etc.
    q_min, q_max = (q[0], q[-1])
    y_ticks = np.linspace(q_min, q_max, 20)  # number of tickmarks
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Q $(Å^{-1})$')
    ax.set_yticks(y_ticks)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylim(q_min, q_max)
    ax.set_title(sample_name)
    plt.savefig(os.path.join(save_path, str(sample_name) + '_GIWAXS_Plot'), dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(1)

    return fig
    
def plotPL(plParams, sampleName, savePath, energyData, timeData, intensityData, intensityDataLog):
    
    if plParams['logplots']:
        fig = plt.figure(figsize=(7, 5))
        plt.contourf(timeData, energyData, intensityDataLog, 20, cmap=plt.cm.jet)
        # Make a colorbar for the ContourSet
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Log-Intensity (a.u.)')
        # adding labels
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (eV)')
        plt.title(str(sampleName) + ' _2D_Plot')
        plt.savefig(os.path.join(savePath, str(sampleName) + '_PL_Plot_Log'), dpi=300, bbox_inches="tight")
        plt.show(block=False)
        plt.pause(1)
    
    fig = plt.figure(figsize=(7, 5))
    plt.contourf(timeData, energyData, intensityData, 20, cmap=plt.cm.jet)
    # Make a colorbar for the ContourSet
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Intensity (a.u.)')
    # adding labels
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (eV)')
    plt.title(str(sampleName) + ' _2D_Plot')
    plt.savefig(os.path.join(savePath, str(sampleName) + '_PL_Plot_Lin'), dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(1)

    
    return fig

def plotIndividually(axisDescription, fileName, sampleName, savePath, xData, timesToPlot, intensityToPlot, timeData):

    fig = plt.figure(figsize=(7, 5))
    
    if len(timesToPlot) < 10:
        for i in range(0,len(timesToPlot)):
           plt.plot(xData, intensityToPlot[i], label = ('{:.2f}'.format((timeData[timesToPlot[i]])) + ' s'))
           plt.legend(loc="upper right")
    else:
        norm = plt.Normalize(timeData[timesToPlot[0]], timeData[timesToPlot[-1]])
        cmap = plt.cm.get_cmap('coolwarm')
        fig, ax = plt.subplots()
        for i in range(len(timesToPlot)):
            ax.plot(xData, intensityToPlot[i], c=cmap(norm(timeData[timesToPlot[i]])))
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), label="Time (s)", ax=fig.gca())
        cbar.set_ticks(np.linspace(timeData[timesToPlot[0]], timeData[timesToPlot[-1]], 11))
        
        
    # adding labels
    plt.xlabel(axisDescription)
    plt.ylabel('Intensity (a.u.)')
    plt.title(str(sampleName) + fileName)
    
    plt.xlim([xData[0], xData[-1]])
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(savePath, str(sampleName) + fileName), dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(1)
    
    return fig


def plotLog(sampleName, savePath, logData, new):
    
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('Time (s)', fontsize = 12)
    ax1.set_ylabel(r'Temperature ($^{\circ}$C)', fontsize = 12, color='r')
    ax1.plot(logData.Time, logData.Pyrometer, 'r-')
    # ax1.set_ylim([0, 105])
    if new:
        ax2 = ax1.twinx()
        ax2.set_ylabel(r'Spin Speed (rpm)', fontsize = 12, color='b')
        ax2.plot(logData.Time, logData.Spin_Motor, 'b-')
    
    plt.title(sampleName + ' Logged Parameters')
    plt.savefig(os.path.join(savePath, str(sampleName) + '_LoggedParameters_Plot'), dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(1)
    
    return fig
    
def plotStacked(genParams, sampleName, savePath, q, timeGIWAXS, intGIWAXS, energyPL, timePL, intPL, logData, logTimeEndIdx):
    
    if genParams['PL']:
        # define subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 9), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})
        
        # PL plot
        # removing negative points from data (important for log plot)
        intPL = np.where(intPL < 1, 1, intPL)
        logIntPL = np.log(intPL)
        i_max = logIntPL.max()
        plt.setp(ax1.get_xticklabels(), fontsize=25)
        cp1 = ax1.contourf(timePL, energyPL, logIntPL/i_max, np.linspace(0/i_max,1, 100),cmap='gist_heat')
        cbax1 = fig.add_axes([0.89, 0.66, 0.03, 0.3])
        cb1 = fig.colorbar(cp1, ax = ax1, cax = cbax1, ticks=np.linspace(0,1,2))
        cb1.set_label(' Norm. Intensity', fontsize = 12, labelpad=-3)
        ax1.set_ylabel('Energy (eV)', fontsize = 12)
    
        # Inset graph for PL plot
        inset = False # set True if zoomed inset is desired, change to False otherwise
        if inset:
            ax_new = plt.axes([.68, .86, .14, .09]) # create inset axes with dimensions [left, bottom, width, height]
            ax_new.contourf(timePL, energyPL, intPL/i_max, np.linspace(0.2/i_max,1, 100), cmap = plt.get_cmap('gist_heat')) # copy code line for larger plot
            # modify tick colors on both axes
            ax_new.tick_params(axis='x', colors='white')
            ax_new.tick_params(axis='y', colors='white')
            # add border around plot - currently need to add an invisible plot to allow connector lines to be added later
            # maybe this could be replaced somehow to improve processing speed/clarity???
            border = plt.axes([.6, .9, .25, .13]) # dimensions of plot border [left, bottom, width, height]
            border.contourf(timePL, energyPL, intPL/i_max, np.linspace(0.2/i_max,1, 100), cmap = plt.get_cmap('gist_heat'), alpha=0)
            # set tick colors and limits for border axes
            border.tick_params(axis='x', colors='none')
            border.tick_params(axis='y', colors='none')
            border.set_xlim(73, 74)
            border.set_ylim(1.55, 1.9)
            # add the actual border
            border.spines['bottom'].set_color('1')
            border.spines['top'].set_color('1')
            border.spines['right'].set_color('1')
            border.spines['left'].set_color('1')
            border.patch.set_facecolor('none')
            border.patch.set_edgecolor("1")
            # set limits for inset axes
            ax_new.set_xlim(72.5,74.5)
            ax_new.set_ylim(1.5,1.9)
            # set tick label spacing
            ax_new.set_yticks(np.arange(1.5,1.9,0.2))
            ax_new.set_xticks(np.arange(73,76,1))
            # add four connector lines between corners of the original and inset plots
            mark_inset(ax1, border, loc1=1, loc2=3, fc="none", ec="1")
            mark_inset(ax1, border, loc1=2, loc2=4, fc="none", ec="1")
            
        giwaxsBarPos = [0.89, 0.3, 0.03, 0.3]
            
    else:
        # define subplots
        fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(6, 5.4), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        giwaxsBarPos = [0.89, 0.425, 0.03, 0.54]

    # GIWAXS plot
    # define ranges and limits
    i_max = intGIWAXS.max()
    i_min = intGIWAXS.min()
    
    cp2 = ax2.contourf(timeGIWAXS, q, intGIWAXS.T/i_max, np.linspace(i_min/i_max, 1, 100), cmap=plt.get_cmap('Greys'))
    cbax2 = fig.add_axes(giwaxsBarPos)
    cb2 = fig.colorbar(cp2, ax = ax2, cax=cbax2, ticks = np.linspace(i_min/i_max, 1, 2))
    cb2.set_label('Norm. Intensity', fontsize = 12, labelpad=-1)
    ax2.set_ylabel(r'q ($Å^{-1}$)', fontsize = 12)

    # Logging plot
    ax3.plot(logData.Time, logData.Pyrometer, 'r-')
    ax3.set_xlabel('Time (s)', fontsize = 12)
    ax3.set_ylabel(r'Temperature ($^{\circ}$C)', fontsize = 12, color='r')
    # ax3.set_ylim([0, 105])
    if not genParams['TempOld']:
        ax4 = ax3.twinx()
        ax4.plot(logData.Time, logData.Spin_Motor, 'b-')
        ax4.set_ylabel(r'Spin speed (rpm)', fontsize = 12, color='b')
        plt.subplots_adjust(right=0.88, top=0.97, bottom = 0.1, hspace=0.1)
    else:
        plt.subplots_adjust(right=0.88, top=0.97, bottom = 0.1, hspace=0.1)

    # General settings for the figure
    ax3.set_xlim(0, logData.Time.iloc[-1]) # assumption that logging was terminated last, change to different axis otherwise
    # supress fig.subtitle(sample_name) if no plot title is desired
    # fig.subtitle(sample_name)
    plt.savefig(os.path.join(savePath, sampleName + '_Stacked_Plot.png'), dpi = 300, bbox_inches = "tight")
    plt.show()
    
    return fig

def htmlPlots(genParams, time_pl, y_pl, z_pl, time_giwaxs, y_giwaxs, z_giwaxs, y_pyro, y_spin, time_log, directory, sample): 

    # set output to static HTML file
    output_file(filename=f'{directory}/{sample}_ALS-data.html', title=f'{sample}_ALS-data')
    
    
    #%%PL-Plot
    if genParams['PL']:
        
        p_pl = figure(
        x_range= (0, time_pl.max()),
        y_range=(y_pl.min(), y_pl.max()),
        width=1200,
        height=640,
        toolbar_location="above")
        
        p_pl.title.text = f'{sample}_heatmap_PL'
        p_pl.xaxis.axis_label = 'Time (s)'
        p_pl.yaxis.axis_label = 'Energy (eV)'
        p_pl.grid.visible = False
        
        color_mapper_pl = LinearColorMapper(palette=Viridis256, low=0, high=z_pl.max())
        plot1 = p_pl.image(image=[z_pl], x=time_pl.min(), y=y_pl.min(), dw=time_pl.max(), dh=y_pl.max()-y_pl.min(), color_mapper=color_mapper_pl, level="image")
        
        numeric_input_pl = NumericInput(value=round(z_pl.max()), low=0, high=round(z_pl.max()), title="Contrast 1", width=70)
        numeric_input_pl.js_link("value", color_mapper_pl, "high")
        
        color_bar_pl = ColorBar(color_mapper=color_mapper_pl, label_standoff=10)
        p_pl.add_layout(color_bar_pl, 'right')
        
        p_pl.add_tools(HoverTool(renderers=[plot1], tooltips=[("Time", "$x"), ("Energy", "$y"), ("Intensity", "@image")]))
        
        tab1 = TabPanel(child=p_pl, title="PL")
        
        if genParams['TempOld']:
            if genParams['Logging']:
                # Setting the second y axis range name and range
                p_pl.extra_y_ranges = {"Temperature (°C)": Range1d(start=0, end=1.05 * y_pyro.max())}
                
                p_pl.add_layout(LinearAxis(y_range_name="Temperature (°C)", axis_label='Temperature (°C)'), 'right')
                temp_pl = p_pl.line(time_log, y_pyro, x="Time", y="Pyrometer", line_width = 1, y_range_name="Temperature (°C)", color="firebrick")
            else:
                temp_pl = []
                speed_pl = []
                numeric_input_pl = []
            
        else:
            if genParams['Logging']:
                # Setting the second y axis range name and range
                p_pl.extra_y_ranges = {"Temperature (°C)": Range1d(start=0, end=1.05 * y_pyro.max()), "Spin Speed (rpm)": Range1d(start=0, end=1.05 * y_spin.max())}
                
                p_pl.add_layout(LinearAxis(y_range_name="Temperature (°C)", axis_label='Temperature (°C)'), 'right')
                temp_pl = p_pl.line(time_log, y_pyro, x="Time", y="Pyrometer", line_width = 1, y_range_name="Temperature (°C)", color="firebrick")
                
                p_pl.add_layout(LinearAxis(y_range_name="Spin Speed (rpm)", axis_label='Spin Speed (rpm)'), 'left')
                speed_pl = p_pl.line(time_log, y_spin, x="Time", y="Spin_Motor", line_width = 1, y_range_name="Spin Speed (rpm)", color="steelblue")
            else:
                temp_pl = []
                speed_pl = []
                numeric_input_pl = []
                
    else:
        p_pl = figure(
        x_range= (0, 1),
        y_range=(0, 1),
        width=1200,
        height=640,
        toolbar_location="above")
        
        p_pl.title.text = 'No PL-data given'
        p_pl.xaxis.axis_label = 'Time (s)'
        p_pl.yaxis.axis_label = 'Energy (eV)'
        p_pl.grid.visible = False
        
        temp_pl = []
        speed_pl = []
        numeric_input_pl = []
        
        tab1 = TabPanel(child=p_pl, title="PL")
        
    
    #%%GIWAXS-Plot
    if genParams['GIWAXS']:

        z_giwaxs = z_giwaxs.T
        
        p_giwaxs = figure(
        x_range= (time_giwaxs.min(), time_giwaxs.max()),
        y_range=(y_giwaxs.min(), y_giwaxs.max()),
        width=1200,
        height=640,
        toolbar_location="above")
        
        p_giwaxs.title.text = f'{sample}_heatmap_GIWAXS'
        p_giwaxs.xaxis.axis_label = 'Time (s)'
        p_giwaxs.yaxis.axis_label = 'q'
        p_giwaxs.grid.visible = False
        
        color_mapper_giwaxs = LinearColorMapper(palette=Greys256, low=0, high=z_giwaxs.max())
        plot2 = p_giwaxs.image(image=[z_giwaxs], x=time_giwaxs.min(), y=y_giwaxs.min(), dw=time_giwaxs.max(), dh=y_giwaxs.max()-y_giwaxs.min(), color_mapper=color_mapper_giwaxs, level="image")
        
        numeric_input_giwaxs = NumericInput(value=round(z_giwaxs.max()), low=0, high=round(z_giwaxs.max()), title="Contrast 2", width=70)
        numeric_input_giwaxs.js_link("value", color_mapper_giwaxs, "high")
        
        color_bar_giwaxs = ColorBar(color_mapper=color_mapper_giwaxs, label_standoff=10)
        p_giwaxs.add_layout(color_bar_giwaxs, 'right')
        
        p_giwaxs.add_tools(HoverTool(renderers=[plot2], tooltips=[("Time", "$x"), ("q", "$y"), ("Intensity", "@image")]))
        
        tab2 = TabPanel(child=p_giwaxs, title="GIWAXS")
        
        # Setting the second y axis range name and range
        if genParams['TempOld']:
            p_giwaxs.extra_y_ranges = {"Temperature (°C)": Range1d(start=0, end=1.05 * y_pyro.max())}
            
            p_giwaxs.add_layout(LinearAxis(y_range_name="Temperature (°C)", axis_label='Temperature (°C)'), 'right')
            temp_giwaxs = p_giwaxs.line(time_log, y_pyro, x="Time", y="Pyrometer", line_width = 1, y_range_name="Temperature (°C)", color="firebrick")
            speed_giwaxs = []
            
        else:
            p_giwaxs.extra_y_ranges = {"Temperature (°C)": Range1d(start=0, end=1.05 * y_pyro.max()), "Spin Speed (rpm)": Range1d(start=0, end=1.05 * y_spin.max())}
            
            p_giwaxs.add_layout(LinearAxis(y_range_name="Temperature (°C)", axis_label='Temperature (°C)'), 'right')
            temp_giwaxs = p_giwaxs.line(time_log, y_pyro, x="Time", y="Pyrometer", line_width = 1, y_range_name="Temperature (°C)", color="firebrick")
            
            p_giwaxs.add_layout(LinearAxis(y_range_name="Spin Speed (rpm)", axis_label='Spin Speed (rpm)'), 'left')
            speed_giwaxs = p_giwaxs.line(time_log, y_spin, x="Time", y="Spin_Motor", line_width = 1, y_range_name="Spin Speed (rpm)", color="steelblue")
            
    else:
        p_giwaxs = figure(
        x_range= (0,1),
        y_range=(0,1),
        width=1200,
        height=640,
        toolbar_location="above")
        
        p_giwaxs.title.text = 'No GIWAXS data given'
        p_giwaxs.xaxis.axis_label = 'Time (s)'
        p_giwaxs.yaxis.axis_label = 'q'
        p_giwaxs.grid.visible = False
        
        temp_giwaxs = []
        speed_giwaxs = []
        numeric_input_giwaxs = []
        
        tab2 = TabPanel(child=p_pl, title="GIWAXS")
    
    
    #%%Logging
    if genParams['Logging']:

        timeLogStart = time_log.max()
        
        if genParams['TempOld']:
            p_log = figure(
            x_range= (0, time_log.max()),
            y_range= (0, 105),#1.05 * y_pyro.max()),
            width=1200,
            height=640,
            toolbar_location="above")
        
            p_log.title.text = f'{sample}_log'
            p_log.xaxis.axis_label = 'Time (s)'
            p_log.yaxis.axis_label = "Temperature (°C)"
            p_log.grid.visible = False
            p_log.line(time_log, y_pyro, x="Time", y="Pyrometer", line_width = 2, y_range_name="Temperature (°C)", color="firebrick")
             
            p_log.add_tools(HoverTool(tooltips=[("Time", "$x"), ("Temperature", "@Pyrometer")]))
            
            tab3 = TabPanel(child=p_log, title="LOG")
        else:
            p_log = figure(
            x_range= (0, time_log.max()),
            y_range= (0, 1.05 * y_spin.max()),
            width=1200,
            height=640,
            toolbar_location="above")
        
            p_log.title.text = f'{sample}_log'
            p_log.xaxis.axis_label = 'Time (s)'
            p_log.yaxis.axis_label = "Spin Speed (rpm)"
            p_log.grid.visible = False
            p_log.line(time_log, y_spin, x="Time", y="Spin_Motor", name = "Time", line_width = 2, color="steelblue")
            
            # Setting the second y axis range name and range
            p_log.extra_y_ranges = {"Temperature (°C)": Range1d(start=0, end=1.05 * y_pyro.max())}
            p_log.add_layout(LinearAxis(y_range_name="Temperature (°C)", axis_label='Temperature (°C)'), 'right')
            
            p_log.line(time_log, y_pyro, x="Time", y="Pyrometer", line_width = 2, y_range_name="Temperature (°C)", color="firebrick")
             
            p_log.add_tools(HoverTool(tooltips=[("Time", "$x"), ("Temperature", "@Pyrometer"), ("Spin speed", "@Spin_Motor")]))
            
            tab3 = TabPanel(child=p_log, title="LOG")
            
    else:
        p_log = figure(
        x_range= (0,1),
        y_range=(0,1),
        width=1200,
        height=640,
        toolbar_location="above")
        
        p_log.title.text = 'No Logging data given'
        p_log.xaxis.axis_label = 'Time (s)'
        p_log.yaxis.axis_label = 'Temperature'
        p_log.grid.visible = False
        
        tab3 = TabPanel(child=p_pl, title="LOG")
        
        timeLogStart = 1
    
    
    #%%
    checkbox = CheckboxGroup(labels=["Temperature", "Spin Speed", "PL peak position"], active=[0,1], width=70)
    
    callback = CustomJS(
    args=dict(
        tpl = temp_pl,
        tgw = temp_giwaxs,
        spl = speed_pl,
        sgw = speed_giwaxs,
        cb = checkbox),
        code="""

    // JavaScript code goes here
    
    tpl.visible = cb.active.includes(0);
    tgw.visible = cb.active.includes(0);
    spl.visible = cb.active.includes(1);
    sgw.visible = cb.active.includes(1);
    """)

    checkbox.js_on_change('active', callback)
    
    numeric_input_start = NumericInput(value=0, low=0, high=round(timeLogStart), title="Start time", width=70)
    numeric_input_start.js_link('value', p_pl.x_range, 'start')
    numeric_input_start.js_link('value', p_giwaxs.x_range, 'start')
    numeric_input_start.js_link('value', p_log.x_range, 'start')
    
    numeric_input_end = NumericInput(value=round(timeLogStart), low=1, high=round(timeLogStart), title="End time", width=70)
    numeric_input_end.js_link('value', p_pl.x_range, 'end')
    numeric_input_end.js_link('value', p_giwaxs.x_range, 'end')
    numeric_input_end.js_link('value', p_log.x_range, 'end')
    
    tabs = Tabs(tabs=[tab1, tab2, tab3])

    full_plot = layout([
        [tabs, [[numeric_input_start],[numeric_input_end],[numeric_input_pl],[numeric_input_giwaxs],checkbox]],
    ])

    # show(full_plot)
    save(full_plot)