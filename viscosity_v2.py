#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:21:09 2023
IMPMC

This script aims to analyse and fit VFT data points acquired with the Linseis L74 PT 1600.
Outputed files should be in the xls format (default from software) to avoid complications such as encoding issues, broken linebreaks, etc.

NB: Angles are given on the outside of the bubble and are thus in clear opposition of what most of the world defines as contact angle but welp just take the complementary (or supplementary) angle, you know just the difference with 180, don't be daft' 

There might be an issue with "special" characters such as µ and °, it remains to be seen but in all honesty, you could just remove preemptively those symbols to prevent any issue

As long as the columns have at least the name of the measurement (i.e. height, volume, area, etc.) the script will find it and assume default units
"""

import pandas as pd
import numpy as np
import os
import scipy as sp
from scipy import optimize as sop
from matplotlib import pyplot as plt
from adjustText import adjust_text
import tkinter as tk
from tkinter import filedialog as fd

def vft(t, a, b, t_o) :
    '''
    Calculate the viscosity at t for a liquid described by the VFT equation of parameters a, b and t_o

    Parameters
    ----------
    t : float
        temperature (in K)
    a : float
        constant.
    b : TYPE
        constant.
    t_o : TYPE
        constant homogenous to K.

    Returns
    -------
    float
        logarithm of the viscosity according to the VFT model

    '''
    return a + b/(t-t_o)

# I had to do it because of course you never know
def K_to_C(theta) :
    return theta-273.15
def C_to_K(temp) :
    return temp+273.15
def identity(theta) :
    return theta
def K_to_F(theta) :
    print('Fuck you\n with a fucking anchor')
    return C_to_F(K_to_C(theta))
def C_to_F(theta) :
    print('Try using an anchor')
    return 9/5 * theta + 32
def F_to_K(theta) :
    return C_to_K(F_to_C(theta))
def F_to_C(theta):
    return 5/9 * (theta-32)    

def temperature_convert(theta, origin, output) :
    '''
    Change the theta temperature given on the origin scale in the output scale

    Parameters
    ----------
    theta : float
        Temperature given in the origin scale.
    origin : str
        K, F or C corresponding to Kelvin, Farenheit or Celsius scale.
    output : str
        K, F or C corresponding to Kelvin, Farenheit or Celsius scale.

    Returns
    -------
    output_temp : float
        temperature in the output scale.

    '''
    switcher = {
        'K': {'K': identity, 'F': K_to_F, 'C': K_to_C},
        'F': {'K': F_to_K, 'F': identity, 'C': F_to_C},
        'C': {'K': C_to_K, 'F': C_to_F, 'C': identity}}
    
    return switcher[origin][output](theta)

def viscosity_graph(sample_id, fixed_points, fixed_viscosity, fitted_data, sphere, fig = None, ax = None) :
    '''
    Generate a graph of both experimental data and fitted data.

    Parameters
    ----------
    sample_id : str
        Name of sample.
    fixed_points : pd.DataFrame
        DESCRIPTION.
    fixed_viscosity : TYPE
        DESCRIPTION.
    fitted_data : TYPE
        DESCRIPTION.
    sphere : bool
        whether or not the sphere point should be taken into account in the calculations.

    Returns
    -------
    None.

    '''
    temp_names = ['FS', 'MS', 'W', 'S', 'HS', 'F']
    names = names = {list(fixed_viscosity.keys())[i]: temp_names[i] for i in range(len(temp_names))}

    points = fixed_viscosity.keys()
    
    data = {'T':[],'log_eta':[], 'label':[]}
    
    for point in points : 
        if fixed_points.loc[sample_id, point] != False :
            data['T'].append(temperature_convert(fixed_points.loc[sample_id, point], 'C', 'K'))
            data['log_eta'].append(fixed_viscosity[point])
            data['label'].append(names[point])
    
    r_squared = fitted_data.loc[sample_id,'R²']
    old_fig = fig
    old_ax = ax
    if fig == None or ax == None :
        fig, ax = plt.subplots(dpi = 360)
        colors = 'red'
        colors2 = 'black'
    else :
        # if more than one figure to draw, it will be ugly because random colors so you know...
        colors = (np.random.random(), np.random.random(), np.random.random())
        colors2 = (np.random.random(), np.random.random(), np.random.random())
    ax.scatter(data['T'], data['log_eta'], color = colors, marker='s', label = f"{sample_id.split('-')[0]} data")
    # Check if there are more than 1 curves to do, that's pretty much the goal of most of the if in that function, added afterwards when I needed many different figures on the same figure
    if old_fig == None or old_ax == None : 
        texts = [plt.text(data['T'][i], data['log_eta'][i], data['label'][i], color = 'red') for i in range(len(data['log_eta']))]
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(r'log $\eta$ (P)')

    fit_temp = np.linspace(data['T'][0], data['T'][-1], 100)
    fit = vft(fit_temp, *fitted_data.loc[sample_id, ['A', 'B', 'T0']])
    A, B, T = fitted_data.loc[sample_id, ['A', 'B', 'T0']]
    ax.plot(fit_temp, fit, color = colors2, linestyle='-.', label = f"fit r² : {r_squared:.4}\n A({A:.02}), B({int(B)}), $T_0$({int(T)}K)")
    ax.legend(fontsize = 8)
    if old_fig == None or old_ax == None : 
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color='k', lw=0.5))
    return fig, ax
    
def viscosity_fit_v2(sample_id, fixed_points, fixed_viscosity, fitted_data, sphere,  DEBUG = False) :
    '''
    

    Parameters
    ----------
    sample_id : str
        Sample ID, always the same, use general caution underlined in other doc
    fixed_points : pandas.DataFrame
        Fixed Points measured out of the other curves and shit
    fixed_viscosity : Dict
        Translation of Fixed points in log viscosity
    fitted_data : pandas.DataFrame
        output, contains all fitted data
    '''
    temp_names = ['FS', 'MS', 'W', 'S', 'HS', 'F']
    names = names = {list(fixed_viscosity.keys())[i]: temp_names[i] for i in range(len(temp_names))}
    initial_values = [-2, 5000, 300] #extracted from pascual article, they are globally in the range of their obtained data
    
    points = fixed_viscosity.keys()
    
    data = {'T':[],'log_eta':[], 'label':[]}
    
    for point in points : 
        if fixed_points.loc[sample_id, point] != False :
            data['T'].append(temperature_convert(fixed_points.loc[sample_id, point], 'C', 'K'))
            data['log_eta'].append(fixed_viscosity[point])
            data['label'].append(names[point])
    if DEBUG : print(data)
    fit_param, pocv = sop.curve_fit(vft, data['T'], data['log_eta'], initial_values)
    
    log_eta = np.array(data['log_eta'])
    modelled = vft(data['T'], *fit_param)
    absError = modelled-log_eta
    rss =  np.sum(np.square(absError))
    tss = np.sum(np.square(log_eta))
    SE = np.square(absError)
    MSE = np.mean(SE)
    RMSE = np.sqrt(MSE)
    r_squared = 1 - (rss / tss)
    
    
    fitted_data.loc[sample_id, 'A'] = fit_param[0]
    fitted_data.loc[sample_id,'B'] = fit_param[1]
    fitted_data.loc[sample_id,'T0'] = fit_param[2]
    fitted_data.loc[sample_id,'R²'] = r_squared
    fitted_data.loc[sample_id, 'RMSE'] = RMSE
    
    print(f'Fit for {sample_id} done:\nR² : {r_squared}\nRMSE : {RMSE}')
    return r_squared


def viscosity_fit(sample_id, fixed_points, fixed_viscosity, fitted_data, sphere,  DEBUG = False) :
    '''
    

    Parameters
    ----------
    sample_id : str
        Sample ID, always the same, use general caution underlined in other doc
    fixed_points : pandas.DataFrame
        Fixed Points measured out of the other curves and shit
    fixed_viscosity : Dict
        Translation of Fixed points in log viscosity
    fitted_data : pandas.DataFrame
        output, contains all fitted data
    '''
    temp_notK = np.array([fixed_points.loc[sample_id, point] for point in fixed_points.keys()])
    temp = []
    for t in temp_notK :
        if t > 25:
            temp.append(temperature_convert(t, 'C', 'K'))
    log_eta = np.array(list(fixed_viscosity.values()))[:len(temp)]
    initial_values = [-2, 5000, 300] #extracted from pascual article, they are globally in the range of their obtained data
    
    names = ['FS', 'MS', 'W', 'S', 'HS', 'F']
    #names = {list(fixed_viscosity.keys())[i]: temp_names[i] for i in range(len(temp_names))}
        
    if DEBUG :

        fig, ax = plt.subplots(dpi = 360)
        ax.scatter(temp, log_eta, color = 'red', marker='s', label = 'data')
        texts = [plt.text(temp[i], log_eta[i], names[i], color = 'red') for i in range(len(temp))]
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel(r'log $\eta$ (P)')
        ax.set_title(sample_id+f" sphere : {sphere}")
        ax.legend()
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color='k', lw=0.5))
    fit_param, pocv = sop.curve_fit(vft, temp, log_eta, initial_values)
    
    modelled = vft(temp, *fit_param)
    absError = modelled-log_eta
    rss =  np.sum(np.square(absError))
    tss = np.sum(np.square(log_eta))
    SE = np.square(absError)
    MSE = np.mean(SE)
    RMSE = np.sqrt(MSE)
    r_squared = 1 - (rss / tss)
    
    
    fitted_data.loc[sample_id, 'A'] = fit_param[0]
    fitted_data.loc[sample_id,'B'] = fit_param[1]
    fitted_data.loc[sample_id,'T0'] = fit_param[2]
    fitted_data.loc[sample_id,'R²'] = r_squared
    fitted_data.loc[sample_id, 'RMSE'] = RMSE
      
    if not DEBUG :
        fig, ax = plt.subplots(dpi = 360)
        ax.scatter(temp, log_eta, color = 'red', marker='s', label = 'data')
        texts = [plt.text(temp[i], log_eta[i], names[i], color = 'red') for i in range(len(temp))]
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel(r'log $\eta$ (P)')
        ax.set_title(sample_id)
    temp = np.linspace(temp[0], temp[-1], 100)
    fit = vft(temp, *fit_param)
    ax.plot(temp, fit, color = 'black', linestyle='-.', label = f'fit r² : {r_squared:.4}')
    ax.legend()
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color='k', lw=0.5))
    print(f'Fit for {sample_id} done:\nR² : {r_squared}\nRMSE : {RMSE}')
    return r_squared

def debug_graphs_weak(sample_id, data, keys, fixed_points, hs_time, s_time) :
    """
    Generates a few graphs to serve as reference, essentially, graphs before the Sphere point, 

    Parameters
    ----------
    sample_id : str
        name of file.
    data : pd.Dataframe
        data read from Linseis output file.
    keys : str list
        should be data.keys()
        
    Returns
    -------
    None. Output graphs

    """
    fig_w, axw1 = plt.subplots(dpi = 360)
    axw2 = axw1.twinx()
    
    axw1.set_title('Data before S point')
    if s_time == False :
        s_time = int(hs_time)
    else :
        s_time = int(s_time)

    max_borne = 300
    axw1.plot(data.loc[:s_time-max_borne, 'Temperature(°C)'], data.loc[: s_time-max_borne, 'ratio H/W'], label = 'ratio H/W', color = 'blue')
    axw1.plot(data.loc[:s_time-max_borne, 'Temperature(°C)'], data.loc[: s_time-max_borne, 'shape(1)'], label = 'shape factor', color = 'green')
    axw1.legend()
    axw1.set_xlabel('temperature')
    axw2.plot(data.loc[:s_time-max_borne, 'Temperature(°C)'], data.loc[:s_time-max_borne, 'contactAngle 1(°)'], label = 'angle 1', color = 'magenta')
    axw2.set_ylabel('Angle (°)')
    axw2.legend()
  
def weaking_point(sample_id, data, keys, fixed_points, ms_time, hs_time, DEBUG = False) :
    """
    Find the flow point as the local maximum of the shape factor curve. As a reminder, the weaking point is defined by the point by which the shape factor as evolved by at least 1.5% AND the tracked angle as evolved by at least 10%

    Parameters
    ----------
    sample_id : str
        name of the sample/file.
    data : pd.Dataframe
        data extracted from the xls data file
    keys : str list
        list of all keys of data.
    fixed_points : pd.Dataframe
        array to store found fixed points
    hs_time : int
        time of the S point, useful to check only points that are after.
    DEBUG : boolean, optional
        Wether or not to show whatever shit this does. Can show graphs, some processes that might cause bugs such as loops and dichotomy process The default is False.

    Returns
    -------
    w_time : int
        time of the Weaking point or deformation point

    """
    # First let's get initial values
    mean_length = 100
    tolerance = 0.005
    value_angle = 0.1
    value_sf = 0.015
    
    # if s_time == False :
    #     s_time = -1
    initial_angle = data.loc[:mean_length, 'contactAngle 1(°)'].mean()
    initial_sf = data.loc[:mean_length, 'shape(1)'].mean()
    if DEBUG : print(initial_angle, '\t', initial_sf)
    # le point que l'on cherche doit se trouver entre le point de max shrinkage et le point de Sphere, sinon il n'est pas là
    relevant_data = data.loc[ms_time:hs_time].copy()
    
    relevant_data['deltaAngle 1'] = abs(relevant_data['contactAngle 1(°)']-initial_angle)/initial_angle
    relevant_data['deltaShape'] = abs(relevant_data['shape(1)']-initial_sf)/initial_sf
    weak_times = relevant_data.query('`deltaAngle 1` > @value_angle - @tolerance and `deltaShape` > @value_sf - @tolerance')['Time(s)'].unique()
    if DEBUG : print(f"found {len(weak_times)} candidates for weaking conditions")
    if len(weak_times) == 0 :
        fixed_points.loc[sample_id,'Deformation'] = False 
        raise IndexError
    
    w_time = weak_times[0]  
    w_temp = data.loc[w_time, 'Temperature(°C)']
    fixed_points.loc[sample_id,'Deformation'] = w_temp    
    return w_time
    
def flow(sample_id, data, keys, fixed_points, hs_height, hs_time, DEBUG = False) :
    """
    Try and find the flow point, of a graph

    Parameters
    ----------
    sample_id : str
        name of the sample/file.
    data : pd.Dataframe
        data extracted from the xls data file
    keys : str list
        list of all keys of data.
    fixed_points : pd.Dataframe
        array to store found fixed points
    hs_height : float64
        height of bubble at HS point, used in the definition of the flow point
    hs_time : int
        time of the HS point, useful to check only points that are after.
    DEBUG : boolean, optional
        Wether or not to show whatever shit this does. Can show graphs, some processes that might cause bugs such as loops and dichotomy process The default is False.

    Returns
    -------
    f_time : int
        time at which the flow point is reached (in seconds)
        
    Raise
    -----
    flowError : error
        Raises this error if no flow point is found because temp is too high

    """
    for key in keys :
        if  'height' in key:
            height_key = key
    tolerance = 0.001 # percent of the values because nothing will be perfect
    # we select only the data after the hs point as anything else is irrelevant, flow point only exist after the thing, and as we cropped any data above the max temp, no issue, everything else is useless
    relevant_data = data.loc[hs_time:].copy()
    relevant_data['height_ratio'] = relevant_data[height_key]/hs_height
    flow_time = relevant_data.query('`height_ratio` > (1-@tolerance)/3 and `height_ratio` < (1+@tolerance)/3')['Time(s)'].unique()
    if len(flow_time) == 0 :
        raise IndexError
    else :
        f_time = flow_time[0]
    f_temp = data.loc[f_time, 'Temperature(°C)']
    fixed_points.loc[sample_id,'Flow'] = f_temp
    return f_time

def shape_points(sample_id, data, keys, fixed_points, sphere, DEBUG = False) :
    """

    This function find the points corresponding to Half-Sphere and Sphere point, it is a walk in the park for the Half-Sphere point, considerably less so for the sphere
    Next is outlined how those points were found

    Parameters
    ----------
    sample_id : str
        name of the sample/file.
    data : pd.Dataframe
        data extracted from the xls data file
    keys : str list
        list of all keys of data.
    fixed_points : pd.Dataframe
        array to store found fixed points
    sphere : bool
        whether to compute the sphere point or not
    DEBUG : boolean, optional
        Wether or not to show whatever shit this does. The default is False.
        
    Returns
    -------
    hs_time : int
        Index of the Half-Sphere Point, also happens to coincide with the acquisition time (in seconds) in the file (if the file acquired data ever second)
        If data was not acquired every second, you can convert using the data, don't be daft.
    s_time : int
        Index of the Sphere Point, also happens to coincide with the acquisition time (in seconds) in the file (if the file acquired data ever second)
        If data was not acquired every second, you can convert using the data, don't be daft..
    hs_height : float64
        height of sample at Half-Sphere point, useful to describe flow

    """
    for key in keys :
        if  'height' in key:
            height_key = key
        elif 'emperat' in key :
            temp_key = key
    # Half-Sphere first, it's easy
    global_max_index = data['ratio H/W'].idxmax(0)
    hs_time = data.query('`ratio H/W` > 0.4985 and `ratio H/W` < 0.5015 and `Time(s)` > @global_max_index')['Time(s)'].unique()[0]
    hs_height = data.loc[hs_time, height_key]
    hs_temp = data.loc[hs_time, temp_key]
    fixed_points.loc[sample_id,'HalfSphere'] = hs_temp
    
    # Sphere point, harder, we'll try to marouflate this shit by cheating. HS is max of shape factor, 
    ## First of all, we want to find the global maxima of the shoulder of H/W ratio just before the shitty HS point, welp, just take the global max, it seems nice
    
    if sphere :
        global_max = data.loc[global_max_index, 'ratio H/W']
        if DEBUG : print(global_max)
    
        ## ths is a remain of when we tried dichotomy to find the local max but honestly, just the global max is enough
        # window = 2000
        # tolerance = 0.001 # percentage around global max
        # dich_data = data.iloc[int(hs_time - window):int(hs_time)].copy()
        # local_max_index = dich_data['ratio H/W'].idxmax(0)
        # if DEBUG : print(local_max_index)
        # local_max = dich_data.loc[local_max_index, 'ratio H/W']
        # flag = 'local'
        # while (local_max >= global_max*(1-tolerance) and local_max <= global_max*(1+tolerance)) or local_max == dich_data.iloc[0]['ratio H/W'] :
        #     if window > 100 : window = window // 2
        #     else : 
        #         flag = 'global'
        #         break
        #     if DEBUG :print(window)
        #     dich_data = data.iloc[int(hs_time - window):int(hs_time)].copy()
        #     local_max_index = dich_data['ratio H/W'].idxmax(0)
        #     local_max = dich_data.loc[local_max_index,'ratio H/W']
        # if DEBUG : flag='global'
        # if flag == 'local' : 
        #     s_time = dich_data.loc[local_max_index,'Time(s)']
        #     s_temp = dich_data.loc[local_max_index, temp_key]
        # elif flag == 'global' :
    
        s_time = data.loc[global_max_index,'Time(s)']
        s_temp = data.loc[global_max_index, temp_key]
    else :
        s_time = False
        s_temp = False
    fixed_points.loc[sample_id,'Sphere'] = s_temp
    
    return hs_time, s_time, hs_height

def shrinkage(sample_id, data, keys, fixed_points, DEBUG = False) :
    """
    Aims to find points of First and Max Shrinkage (FS and MS) which corresponds to 10 and 90% of contraction first seen in data.
    It reads data and modifies fixed_points with the points it found

    Parameters
    ----------
    sample_id : str
        name of the sample/file.
    data : pd.Dataframe
        data extracted from the xls data file
    keys : str list
        list of all keys of data.
    fixed_points : pd.Dataframe
        array to store found fixed points
    DEBUG : boolean, optional
        Wether or not to show whatever shit this does. The default is False.

    Returns
    -------
    fs_time : int
        Index of the data of first Shrinkage, also happens to coincide with the acquisition time (in seconds) in the file (if the file acquired data ever second)
        If data was not acquired every second, you can convert using the data, don't be daft
    ms_time : int
        Same as fs_time but for Max Shrinkage

    """
    
    # first the data is truncated to remove points that are too far down the line and will serve no use.
    # /!\ purely empirical condition of cutting
    relevant_max = data['ratio H/W'].idxmax(0)
    relevant_data = data.loc[:relevant_max, :].copy()
    
    if DEBUG : debug_fig_d, ax1_d = plt.subplots(dpi=360)
    time = np.array(relevant_data['Time(s)'])
    # on lisse les courbes 
    density = sp.signal.savgol_filter(relevant_data['density(/mm³)'], 20, 2)
    if DEBUG : ax1_d.plot(time, density, color='black', linestyle='-.', label = 'Relative Density') 
    #on derive une fois
    diff_t = np.gradient(time)
    diff_d = np.gradient(density)
    deriv= sp.signal.savgol_filter(diff_d/diff_t, 100, 1)
    
    if DEBUG : 
        ax2_d = ax1_d.twinx()
        ax2_d.plot(time, deriv, linestyle='-', label = 'Derivative')
    
    # on trouve le premier max de la derivée
    indice_diff_max = sp.signal.find_peaks(deriv,height=max(deriv)/10, width=(10,1000))
    max_diff = np.argmax(deriv)
    flag = False
    for pos, ind in enumerate(indice_diff_max[0]) :
        if ind > max_diff * 0.99 and ind < max_diff * 1.01 :
            flag = True
            peak_number = pos
            peak_time = time[ind]
    if not flag :
        peak_time = time[indice_diff_max[0]][0]
        peak_number = 0        
    if DEBUG :  ax2_d.vlines(peak_time, ymin = ax2_d.get_ylim()[0], ymax = max(deriv), color = 'magenta')
    test = sp.signal.peak_prominences(deriv, indice_diff_max[0])
    width, pro, left, right = sp.signal.peak_widths(deriv, indice_diff_max[0], rel_height = 0.25, prominence_data=test)
    
    delta_left = int(peak_time - left[peak_number])
    delta_right = int(right[peak_number]-peak_time)
    
    
    min_left = int(left[peak_number]-15*delta_left)
    max_left = int(left[peak_number]-10*delta_left)
    min_right = int(right[peak_number]+3*delta_right)
    max_right = int(right[peak_number]+5*delta_right)
    
    density_ini = np.mean(density[min_left: max_left])
    density_fin = np.mean(density[min_right: max_right])
    
    delta_den = density_fin - density_ini
    
    if DEBUG :
        ax1_d.hlines(density_ini, min_left, max_left, color='green', linestyle='solid', linewidths = 10)
        ax1_d.hlines(density_fin, min_right,max_right, color='green', linestyle='solid', linewidths = 10)
        ax1_d.set_title(sample_id)
    
    density_redux = density[min_left: max_right]
    fs_indexes = np.where(np.logical_and(density_redux >= (density_ini + 0.099*delta_den), density_redux <= (density_ini + 0.101*delta_den)))
    ms_indexes = np.where(np.logical_and(density_redux >= (density_ini + 0.899*delta_den), density_redux <= (density_ini + 0.901*delta_den)))     
    
    try : 
        fs_index =  min_left + fs_indexes[0][len(fs_indexes)//2]
    except:
        fs_index = 0
        print('No candidate for First Shrinkage')
    try : 
        ms_index = min_left + ms_indexes[0][len(ms_indexes)//2]
    except:
        ms_index = 0
        print('No candidate for Max Shrinkage')
    
    fs_time = data.iloc[fs_index]['Time(s)']
    ms_time = data.iloc[ms_index]['Time(s)']
    
    # last_time = data.loc[max_right, 'Time(s)']
    if DEBUG :
        ymin = ax1_d.get_ylim()[0]
        ymax = ax1_d.get_ylim()[1]
        ax1_d.vlines(fs_time, ymin = ymin, ymax = ymax, color = 'orange')
        ax1_d.vlines(ms_time, ymin = ymin, ymax = ymax, color = 'orange')
    fs_temp = data.loc[fs_index, 'Temperature(°C)']
    ms_temp = data.loc[ms_index, 'Temperature(°C)']
    fixed_points.loc[sample_id,'FS'] = fs_temp
    fixed_points.loc[sample_id,'MS'] = ms_temp
    
    return fs_time, ms_time

def debug_graphs_shape(sample_id, data, keys, fixed_points, hs_time, s_time) :
    """
    Generates a few graphs to serve as reference, essentially, graphs around the HalfSphere point, 

    Parameters
    ----------
    sample_id : str
        name of file.
    data : pd.Dataframe
        data read from Linseis output file.
    keys : str list
        should be data.keys()
        
    Returns
    -------
    None. Output graphs

    """
    fig, ax1 = plt.subplots(dpi = 360)
    ax2 = ax1.twinx()
    
    ax1.set_title('ratio around HS point')
    hs_time = int(hs_time)
    min_borne = 4000
    max_borne = 300
    ax1.plot(data.loc[hs_time-min_borne: hs_time+max_borne, 'Temperature(°C)'], data.loc[hs_time-min_borne: hs_time+max_borne, 'ratio H/W'], label = 'ratio H/W', color = 'blue')
    ax1.plot(data.loc[hs_time-min_borne: hs_time+max_borne, 'Temperature(°C)'], data.loc[hs_time-min_borne: hs_time+max_borne, 'shape(1)'], label = 'shape factor', color = 'green')
    ax1.legend()
    ax1.set_xlabel('temperature')
    ymin = ax1.get_ylim()[0]
    ymax = ax1.get_ylim()[1]
    if s_time : ax1.vlines(data.loc[s_time, 'Temperature(°C)'], ymin = ymin, ymax = ymax, color = 'orange')
    ax1.vlines(data.loc[hs_time, 'Temperature(°C)'], ymin = ymin, ymax = ymax, color = 'red')
    
    ax2.plot(data.loc[hs_time-min_borne: hs_time+max_borne, 'Temperature(°C)'], data.loc[hs_time-min_borne: hs_time+max_borne, 'contactAngle 1(°)'], label = 'angle 1', color = 'magenta')
    ax2.set_ylabel('Angle (°)')
    ax2.legend()

def general_graphs(sample_id, data, keys) :
    """
    Outputs a few graphs to check on things

    Parameters
    ----------
    sample_id : str
        id of the sample currently being processed.
    data : pd.Dataframe
        Data
    keys : str list
        all keys of data.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(dpi = 360)
    #ax.set_title(f"General Graphs of {sample_id}")
    
    # let's compute ratio and shape factor

    ax2 = ax.twinx()
    ax2.plot(data['Temperature(°C)'], data['contactAngle 1(°)'], label = 'angle 1', linestyle = '--', color = 'magenta')
    ax2.plot(data['Temperature(°C)'], data['contactAngle 2(°)'], label = 'angle 2', linestyle = '--', color = 'pink')
    ax.plot(data['Temperature(°C)'], data['ratio H/W'], label = 'ratio H/W', color = 'blue')
    ax.plot(data['Temperature(°C)'], data['shape(1)'], label = 'shape', color = 'green')
    
    ax.legend()
    ax2.legend()
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('ratio and shape')
    ax2.set_ylabel('Angle (°)')
    
def file_reader(file_path) :
    """
    Read files from file path and prepare them for processing

    Parameters
    ----------
    file_path : str
        Path to the file, might be absolute, might be relative, don't care.

    Returns
    -------
    sample_id : str
        id of the sample.
    data : pd.Dataframe
        data extracted from file located at <file_path>, with every point after max temp removed
    data.keys() : str list
        keys from data

    """
    # Reading file
    new_path = os.path.abspath(file_path)
    if os.path.isfile(new_path):
        full_data = pd.read_excel(io = f'{file_path}', skiprows=[0,1,2], false_values=['FALSE'])
        sample_id = os.path.split(file_path)[1].split('.')[0]
    else :
        pass
    
    ## Account for error in key name
    for key in full_data.keys() :
        if 'width' in key:
            width_key = key
        elif 'height' in key:
            height_key = key
    # let's get the max and burn everything else
    index_max = full_data['Temperature(°C)'].idxmax(0) 
    data = full_data.iloc[:index_max,:].copy()
    # adding a few columns
    data.loc[:, 'density(/mm³)'] = 0.001*data.loc[:, f'{height_key}']/(data.loc[:, 'area(mm²)']*data.loc[:,'area(mm²)'])
    data.loc[:, 'ratio H/W'] = data.loc[:, f'{height_key}']/data.loc[:, f'{width_key}']   
    
    return sample_id, data, data.keys()    
    
    
def visco(file_path, fixed_points, fitting_data, fixed_viscosity, DEBUG=False, fig = None, ax = None) :
    sample_id, data, keys = file_reader(file_path)
    sphere = False
    if DEBUG : general_graphs(sample_id, data, data.keys())
    
    fs_time, ms_time = shrinkage(sample_id, data, data.keys(), fixed_points, DEBUG = DEBUG)
    hs_time, s_time, hs_height = shape_points(sample_id, data, data.keys(), fixed_points, sphere = sphere, DEBUG = DEBUG)
    if DEBUG : debug_graphs_shape(sample_id, data, data.keys(), fixed_points, hs_time, s_time)
    try : 
        f_time = flow(sample_id, data, data.keys(), fixed_points, hs_height, hs_time, DEBUG = DEBUG)
    except IndexError: 
        print('Flow point is above 1500°C') 
    except : 
        print('Something else went wrong while looking for flow point')
    if DEBUG : debug_graphs_weak(sample_id, data, data.keys(), fixed_points, hs_time, s_time)
    try :
        w_point = weaking_point(sample_id, data, data.keys(), fixed_points, ms_time, hs_time, DEBUG = DEBUG)
    except IndexError :
        print('No Weaking point detected')
    
    # now we'll test on sphere point
    r_sq_1 = viscosity_fit_v2(sample_id, fixed_points, fixed_viscosity, fitting_data, sphere, DEBUG = DEBUG)
    if DEBUG : viscosity_graph(sample_id, fixed_points, fixed_viscosity, fitting_data, sphere)
    sphere = True
    hs_time, s_time, hs_height = shape_points(sample_id, data, data.keys(), fixed_points, sphere = sphere, DEBUG = DEBUG)
    r_sq_2 = viscosity_fit_v2(sample_id, fixed_points, fixed_viscosity, fitting_data, sphere, DEBUG = DEBUG)
    if DEBUG : viscosity_graph(sample_id, fixed_points, fixed_viscosity, fitting_data, sphere)
    if r_sq_1 > r_sq_2 and abs(r_sq_1 - r_sq_2) > 1e-4:
        sphere = False
        hs_time, s_time, hs_height = shape_points(sample_id, data, data.keys(), fixed_points, sphere = sphere, DEBUG = DEBUG)
        viscosity_fit_v2(sample_id, fixed_points, fixed_viscosity, fitting_data, sphere, DEBUG = DEBUG)
        print(f"Sphere point for {sample_id} is found and helpful")
    else :
        print(f"Sphere point for {sample_id} was not helpful and was removed")
    if not DEBUG : fig, ax = viscosity_graph(sample_id, fixed_points, fixed_viscosity, fitting_data, sphere, fig = fig, ax = ax)
    return fig, ax

def file_fetcher() :
    """
    It's pretty obvious in the name isn't it ? It fetches files... 

    Returns
    -------
    files : list str
        A list of paths that points to xls files

    """
    path = '/media/lost'
    shell = tk.Tk()
    files = fd.askopenfilenames(initialdir= path, filetypes=[("Excel Like", "*.xls"), ("Comma Separated Values", "*.csv")])
    shell.destroy()  
    return files

def init(files) :
    """
    Generate a few thingies that are useful down the line

    Parameters
    ----------
    files : str list
        name of the files and their path

    Returns
    -------
    fitting_data : pd.Dataframe
        table with all fit relevant information.
    fixed_points : pd.Dataframe
        All found fixed points.
    fixed_viscosity : dict
        correspondance between names of fixed points and viscosity value.

    """
    names = [os.path.split(file)[1].split('.')[0] for file in files]
    # Names of fixed viscosity points
    points = ['FS', 'MS', 'Deformation', 'Sphere', 'HalfSphere', 'Flow']
    # log of fixed viscosity values
    values = [9.1, 7.8, 6.3, 5.4, 4.1, 3.4]
    # Let's make a Dict out of that to have a translation
    fixed_viscosity = {name: values[index] for index, name in enumerate(points)}
    # Name of fitting parameters
    fitting_parameters = ['A', 'B', 'T0', 'R²', 'RMSE']
    
    # another dataframe that will hold the temperature of certain fixed data points for all samples
    fixed_points = pd.DataFrame(data=False, columns=points, index=names)
    # a last dataframe to hold the fitting obtained
    fitting_data = pd.DataFrame(data=False, columns=fitting_parameters, index=names)
    return fitting_data, fixed_points, fixed_viscosity

def main(DEBUG=False) :
    files = file_fetcher()
    fitting_data, fixed_points, fixed_viscosity = init(files)
    i = 0
    for file in files :
        sample_id = os.path.split(file)[1].split('.')[0]
        print(f"Beginning treatment for {sample_id}")
        if i == 0 :
            ax = None
            fig = None
        fig, ax = visco(file, fixed_points, fitting_data, fixed_viscosity, DEBUG = DEBUG, fig = fig, ax = ax)
        i += 1
    print(fixed_points)
    print(fitting_data)

def debug(DEBUG = True) :
    files = file_fetcher()
    for file in files :
        sample_id, data, keys = file_reader(file)
        general_graphs(sample_id, data, keys)