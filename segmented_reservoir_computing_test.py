# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:21:22 2023

@author: AGCRAIG
"""

import numpy as np
import matplotlib.pyplot as plt
import time as time
from reservoirml import IzhikevichReservoirComputerSegmented

code_start_time = time.time()

num_brain_areas = 360
neurons_per_area = 360
data_time_resolution = 0.72
selected_area = 0
num_training_reps = 3

time_series_file_name = 'E:\\HCP_data\\fMRI_binaries\\ts_100206_1_LR.bin'
ts_data = np.fromfile(time_series_file_name, np.float64)
num_data_time_points = len(ts_data)//num_brain_areas
data_ts = np.reshape( ts_data, (num_data_time_points, num_brain_areas, 1) ).transpose( (1, 0 , 2) )
data_times = np.linspace(data_time_resolution, data_time_resolution*num_data_time_points, num_data_time_points)
print('Loaded data time series with size')
print( np.shape(data_ts) )

anatomy_file_name = 'E:\\HCP_data\\anatomy_binaries\\anatomy_100206.bin'
anatomy_data = np.fromfile(anatomy_file_name, np.float64)
anatomy_features_per_area = len(anatomy_data)//num_brain_areas
anatomy_consts = np.reshape( anatomy_data, (anatomy_features_per_area, num_brain_areas) ).transpose()
print('Loaded anatomy constants with size')
print( np.shape(anatomy_consts) )

all_to_all_model = IzhikevichReservoirComputerSegmented(num_brain_areas, neurons_per_area, anatomy_features_per_area)
all_to_all_model.set_const_input(anatomy_consts)

sim_ts = np.zeros( np.shape(data_ts) )
sim_ts[:,0,:] = data_ts[:,0,:]

print_every_seconds = 60
last_print_time = code_start_time
print('starting training')
for rep_index in range(num_training_reps):
    all_to_all_model.reset()
    for data_time_step in range(1,num_data_time_points):
        all_to_all_model.sim_step()
        all_to_all_model.rls_step( data_ts[:,data_time_step,:] )
        current_time = time.time()
        if current_time - last_print_time >= print_every_seconds:
            print('time', current_time - code_start_time, 'seconds, rep', rep_index, ', training step', data_time_step)
            last_print_time = current_time
    all_to_all_model.reset()
    for sim_time_step in range(1, num_data_time_points):
        all_to_all_model.sim_step()
        sim_ts[:,sim_time_step,:] = all_to_all_model.prediction[:,0,:]
        current_time = time.time()
        if current_time - last_print_time >= print_every_seconds:
            print('time', current_time - code_start_time, 'seconds, rep', rep_index, ', validation step', data_time_step)
            last_print_time = current_time
    rmse = np.sqrt( np.mean( np.square(sim_ts[:,1:,:] - data_ts[:,1:,:]) ) )
    print('rep', rep_index, 'RMSE', rmse)
    plt.plot( data_times, data_ts[selected_area,:,:], '-r', data_times, sim_ts[selected_area,:,:], '--g' )
    plt.xlabel('time (seconds)')
    plt.ylabel('BOLD signal')
    plt.draw()
print( 'done, time', time.time() - code_start_time )