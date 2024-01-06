# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:21:22 2023

@author: AGCRAIG
"""

import argparse
import torch
# import numpy as np
import time as time
import csv
import os
import pickle
import hcpdatautils as hcp
from reservoirmltorch import IzhikevichReservoirComputerSegmented
 
code_start_time = time.time()
print('started', __file__)

# Adapted from https://docs.python.org/3/howto/argparse.html#id1
parser = argparse.ArgumentParser(description="Train a segmented reservoir computing model on one sequence. Then see how fast it forgets when trained on another.")
parser.add_argument("-a", "--subject_id_1", type=str, default="100206", help="ID of HCP subject") 
parser.add_argument("-b", "--time_series_suffix_1", type=str, default="1_LR", help="suffix indiccating which of the 4 time series for the subject to use") 
parser.add_argument("-c", "--subject_id_2", type=str, default="516742", help="ID of HCP subject") 
parser.add_argument("-d", "--time_series_suffix_2", type=str, default="1_LR", help="suffix indiccating which of the 4 time series for the subject to use") 
parser.add_argument("-e", "--data_directory", type=str, default="data", help="base directory under which to find fMRI_ts_binaries and anatomy_binaries")
parser.add_argument("-f", "--simulated_time_series_directory", type=str, default="results/sim_ts_binaries", help="directory in which to save binaries of the simulated time series")
parser.add_argument("-g", "--trained_model_directory", type=str, default="trained_models", help="directory in which to save pickle dump of trained model")
parser.add_argument("-i", "--training_log_directory", type=str, default="results/rmse_tables", help="path to which to save the training log CSV file")
parser.add_argument("-j", "--neurons_per_segment", type=int, default=1000, help="neurons per segment")
parser.add_argument("-k", "--reservoir_density", type=float, default=0.1, help="reservoir connection density")
parser.add_argument("-l", "--intra_reservoir_weight", type=float, default=5000.0, help="intra-reservoir connection weighting factor")
parser.add_argument("-m", "--inter_reservoir_weight", type=float, default=5000.0, help="inter-reservoir connection weighting factor")
parser.add_argument("-n", "--const_input_weight", type=float, default=5000.0, help="anatomy input weighting factor")
parser.add_argument("-o", "--prediction_input_weight", type=float, default=5000.0, help="previous prediction as input weighting factor")
parser.add_argument("-p", "--device_name", type=str, default="cuda", help="device name")   
parser.add_argument("-q", "--training_reps_per_time_series", type=int, default=10, help="number of training reps for each time series")
parser.add_argument("-s", "--rls_steps_per_data_step", type=int, default=1, help="recursive least squares steps per data time point") 
parser.add_argument("-t", "--sim_steps_per_rls_step", type=int, default=1, help="sim steps per RLS step") 
parser.add_argument("-u", "--sim_steps_before_training", type=int, default=0, help="sim steps to run before training starts") 
parser.add_argument("-v", "--normalization_mode", type=str, default="std-mean", help="method by which to normalize the data, either std-mean, min-max, or none") 
parser.add_argument("-w", "--dt", type=float, default=0.04, help="Euler integration step size")
parser.add_argument("-x", "--normalize_const_input", type=bool, default=False, help="Euler integration step size")
args = parser.parse_args()

subject_id_1 = args.subject_id_1
time_series_suffix_1 = args.time_series_suffix_1
subject_id_2 = args.subject_id_2
time_series_suffix_2 = args.time_series_suffix_2
data_directory = args.data_directory
simulated_time_series_directory = args.simulated_time_series_directory
trained_model_directory = args.trained_model_directory
training_log_directory = args.training_log_directory
neurons_per_segment = args.neurons_per_segment
reservoir_density = args.reservoir_density
intra_reservoir_weight = args.intra_reservoir_weight
inter_reservoir_weight = args.inter_reservoir_weight
const_input_weight = args.const_input_weight
prediction_input_weight = args.prediction_input_weight
device_name = args.device_name
training_reps_per_time_series = args.training_reps_per_time_series
rls_steps_per_data_step = args.rls_steps_per_data_step
sim_steps_per_rls_step = args.sim_steps_per_rls_step
sim_steps_before_training = args.sim_steps_before_training
normalization_mode = args.normalization_mode
dt = args.dt
normalize_const_input = args.normalize_const_input

sim_steps_per_data_step = sim_steps_per_rls_step * rls_steps_per_data_step

time_series_name_1 = f"{subject_id_1}_{time_series_suffix_1}"
time_series_name_2 = f"{subject_id_2}_{time_series_suffix_2}"
const_input_file_path_1 = hcp.get_area_features_file_path(data_directory, subject_id_1)
const_input_file_path_2 = hcp.get_area_features_file_path(data_directory, subject_id_2)
data_time_series_file_path_1 = hcp.get_time_series_file_path(data_directory, subject_id_1, time_series_suffix_1)
data_time_series_file_path_2 = hcp.get_time_series_file_path(data_directory, subject_id_2, time_series_suffix_2)
run_properties_string = f"interleaved_{time_series_name_1}_and_{time_series_name_2}_reps_{training_reps_per_time_series}_pre_training_{sim_steps_before_training}_rls_per_data_{rls_steps_per_data_step}_sim_per_rls_{sim_steps_per_rls_step}_nps_{neurons_per_segment}_Q_const_{const_input_weight}_Q_pred_{prediction_input_weight}_G_inter_{inter_reservoir_weight}_norm_{normalization_mode}_dt_{dt}_const_norm_{normalize_const_input}"
training_log_file_path = os.path.join( training_log_directory, f"rmse_table_{run_properties_string}.csv" )
trained_model_file_path = os.path.join( trained_model_directory, f"model_{run_properties_string}.bin")

device = torch.device(device_name)

print('loading training data...')
if normalize_const_input:
    const_means = hcp.load_matrix_from_binary( hcp.get_area_feature_means_file_path(data_directory), device=device )
    const_stds = hcp.load_matrix_from_binary( hcp.get_area_feature_stds_file_path(data_directory), device=device )
    print('loaded anatomy feature means and stds for normalization')
    const_input_1 = ( hcp.load_matrix_from_binary(const_input_file_path_1, device=device) - const_means )/const_stds
    print(f'Loaded const input data from {const_input_file_path_1} with size {const_input_1.size()}')
    const_input_2 = ( hcp.load_matrix_from_binary(const_input_file_path_2, device=device) - const_means )/const_stds
    print(f'Loaded const input data from {const_input_file_path_2} with size {const_input_2.size()}')
else:
    const_input_1 = hcp.load_matrix_from_binary(const_input_file_path_1, device=device)
    print(f'Loaded const input data from {const_input_file_path_1} with size {const_input_1.size()}')
    const_input_2 = hcp.load_matrix_from_binary(const_input_file_path_2, device=device)
    print(f'Loaded const input data from {const_input_file_path_2} with size {const_input_2.size()}')
data_ts_1 = hcp.load_time_series_from_binary(data_time_series_file_path_1, device=device, normalize='std-mean')
print(f'Loaded data time series from {data_time_series_file_path_1} with size {data_ts_1.size()}')
data_ts_2 = hcp.load_time_series_from_binary(data_time_series_file_path_2, device=device, normalize='std-mean')
print(f'Loaded data time series from {data_time_series_file_path_2} with size {data_ts_2.size()}')

num_segments = const_input_1.size(0)
const_inputs_per_segment = const_input_1.size(1)
print('initialized model...')
all_to_all_model = IzhikevichReservoirComputerSegmented( \
    dt=dt, \
    num_segments=num_segments, \
    neurons_per_segment=neurons_per_segment, \
    const_inputs_per_segment=const_inputs_per_segment, \
    reservoir_density=reservoir_density, \
    G_intra=intra_reservoir_weight, \
    G_inter=inter_reservoir_weight, \
    Q_const=const_input_weight, \
    Q_prediction=prediction_input_weight, \
    device=device \
)

# Train on the second time series the same way,
# but also check how much worse the RMSE for the first time series gets.

sim_steps_per_data_step = sim_steps_per_rls_step * rls_steps_per_data_step

num_data_time_points_1 = data_ts_1.size(1)
data_fc_1 = hcp.get_fc(data_ts_1)
data_ps_1 = hcp.get_ps(data_ts_1)
data_ts_init_1 = data_ts_1[:,0]

num_data_time_points_2 = data_ts_2.size(1)
data_fc_2 = hcp.get_fc(data_ts_2)
data_ps_2 = hcp.get_ps(data_ts_2)
data_ts_init_2 = data_ts_2[:,0]

print('starting training...')
with open(training_log_file_path, 'w', encoding='UTF8') as training_log_file:
    print(training_log_file_path)
    training_log_writer = csv.writer(training_log_file)
    training_log_writer.writerow(['rep', 'time', 'rmse_1', 'rmse_ps_1', 'rmse_fc_1', 'rmse_2', 'rmse_ps_2', 'rmse_fc_2'])
    
    for rep_index in range(training_reps_per_time_series):

        print(f"training rep {rep_index}")
        all_to_all_model.set_const_input(const_input_1)
        all_to_all_model.train_once(data_ts_1, sim_steps_before_training, rls_steps_per_data_step, sim_steps_per_rls_step)
        print(f"validation rep {rep_index} on ts 1")
        sim_ts_1 = all_to_all_model.predict(data_ts_init_1, num_data_time_points_1, sim_steps_per_data_step, sim_steps_before_training)
        print('saving simulated time series 1...')
        sim_time_series_file_path_1 = os.path.join(simulated_time_series_directory, f"ts_{time_series_name_1}_{run_properties_string}_rep_{rep_index}.pt")
        torch.save(sim_ts_1, sim_time_series_file_path_1)
        rmse_1 = hcp.get_ts_rmse_torch(sim_ts_1, data_ts_1)
        sim_ps_1 = hcp.get_ps(sim_ts_1)
        rmse_ps_1 = hcp.get_ps_rmse_torch(sim_ps_1, data_ps_1)
        sim_fc_1 = hcp.get_fc(sim_ts_1)
        rmse_fc_1 = hcp.get_fc_rmse_torch(sim_fc_1, data_fc_1)
        time_since_start = time.time() - code_start_time
        
        print(f"training rep {rep_index}")
        all_to_all_model.set_const_input(const_input_2)
        all_to_all_model.train_once(data_ts_2, sim_steps_before_training, rls_steps_per_data_step, sim_steps_per_rls_step)
        print(f"validation rep {rep_index} on ts 2")
        sim_ts_2 = all_to_all_model.predict(data_ts_init_2, num_data_time_points_2, sim_steps_per_data_step, sim_steps_before_training)
        print('saving simulated time series 2...')
        sim_time_series_file_path_2 = os.path.join(simulated_time_series_directory, f"ts_{time_series_name_2}_{run_properties_string}_rep_{rep_index}.pt")
        torch.save(sim_ts_2, sim_time_series_file_path_2)
        rmse_2 = hcp.get_ts_rmse_torch(sim_ts_2, data_ts_2)
        sim_ps_2 = hcp.get_ps(sim_ts_2)
        rmse_ps_2 = hcp.get_ps_rmse_torch(sim_ps_2, data_ps_2)
        sim_fc_2 = hcp.get_fc(sim_ts_2)
        rmse_fc_2 = hcp.get_fc_rmse_torch(sim_fc_2, data_fc_2)
        time_since_start = time.time() - code_start_time

        training_log_writer.writerow([rep_index, time_since_start, rmse_1, rmse_ps_1, rmse_fc_1, rmse_2, rmse_ps_2, rmse_fc_2])
        print(f'time {time_since_start:.1f} seconds, rep {rep_index}, ts 1 RMSE {rmse_1:.3f}, ts 1 PS RMSE {rmse_ps_1:.3f}, ts 1 FC RMSE {rmse_fc_1:.3f}, ts 2 RMSE {rmse_2:.3f}, ts 2 PS RMSE {rmse_ps_2:.3f}, ts 2 FC RMSE {rmse_fc_2:.3f}')

    print(f'done training, time {time.time() - code_start_time:.1f}')
    print('saving model...')
    model_file = open(trained_model_file_path, 'wb')
    pickle.dump(all_to_all_model, model_file)
    model_file.close()
    print(f'done, time {time.time() - code_start_time:.1f}')
