#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:14:28 2023

@author: lost
"""

import pandas as pd
import numpy as np
import os
import scipy as sp
from scipy import optimize as sop
from matplotlib import pyplot as plt
from adjustText import adjust_text
import os
# import shutil
import tkinter as tk
from tkinter import filedialog as fd

colors = "bgrcmy"

path = '/media/lost'
shell = tk.Tk()
files = fd.askopenfilenames(initialdir= path, filetypes=[("Excel Like", "*.xls"), ("Comma Separated Values", "*.csv")])
shell.destroy() 

fig, ax = plt.subplots(dpi=360)
ins = ax.inset_axes([0.1, 0.1, 0.4, 0.4])
i = 1
for file in files :
    if os.path.isfile(file):
        full_data = pd.read_excel(io = f'{file}', skiprows=[0,1,2], false_values=['FALSE'])
    else :
        print('HOW THE HELL DID YOU REACH HERE')
    sample_id = os.path.split(file)[1][:-4]
    index_max = full_data['Temperature(°C)'].idxmax(0) 
    last_index = full_data['Time(s)'].idxmax(0)
    if last_index < index_max + 200 :
        data = full_data.iloc[:last_index,:] 
    else :
        data = full_data.iloc[:index_max+500,:]    
    
    maxi = data['contactAngle 1(°)'].idxmax(0)
    #temp = [(key, float(consigne_temp[key][sample_id])) for key in consigne_temp.keys() if 'T' in key and not consigne_temp[key][sample_id] == False]
    ax.plot(data['Temperature(°C)'], data['contactAngle 1(°)'], linestyle='-', label = f'{sample_id}', color = colors[i%len(colors)])   
    #ax.plot(data['Temperature(°C)'], data['contactAngle 2(°)'], linestyle='-', label = f'{sample_id} droite', color = colors[i%len(colors)])   


    ins.plot(data['Temperature(°C)'], data['contactAngle 1(°)'], linestyle='-', label = f'{sample_id}', color = colors[i%len(colors)])   
    #ins.plot(data['Temperature(°C)'], data['contactAngle 2(°)'], linestyle='-', label = f'{sample_id} droite', color = colors[i%len(colors)])

    
    i+=1
ins.set_xlim(1000, 1200)
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Angle (°)')
ax.legend(fontsize = 6)