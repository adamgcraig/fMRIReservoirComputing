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
import hcpdatautils as hcp
from izrcml import IzhikevichStructuredReservoirComputer
 
code_start_time = time.time()
print('started', __file__)

# Adapted from https://docs.python.org/3/howto/argparse.html#id1
parser = argparse.ArgumentParser(description="Train reservoir computing models on a series of time series inputs with corresponding constant inputs.")
parser.add_argument("-a", "--data_directory", type=str, default="data", help="list of time series and constant feature file path pairs")
parser.add_argument("-b", "--subject_list_file", type=str, default="training_subject_ids.txt", help="list of subject IDs")
parser.add_argument("-c", "--max_num_subjects", type=int, default=1000, help="maximum number of subjects on which to train")
parser.add_argument("-d", "--simulated_time_series_directory_path", type=str, default="simulated_time_series.bin", help="path at which to save binary time series of prediction outputs")
parser.add_argument("-e", "--training_log_file_path", type=str, default="training_log.csv", help="path to which to save the training log CSV file")
parser.add_argument("-f", "--sim_steps_before_training", type=int, default=0, help="sim steps to run before training starts")
parser.add_argument("-g", "--training_reps_per_time_series", type=int, default=10, help="number of training reps per time series")
parser.add_argument("-i", "--rls_steps_per_data_step", type=int, default=1, help="recursive least squares steps per data time point")
parser.add_argument("-j", "--sim_steps_per_rls_step", type=int, default=1, help="sim steps per RLS step")
parser.add_argument("-k", "--num_communities", type=int, default=360, help="number of communities of neurons, each with one output")
parser.add_argument("-l", "--neurons_per_community", type=int, default=2000, help="neurons per community")
parser.add_argument("-m", "--const_inputs_per_community", type=int, default=4, help="number of constant inputs per community")
parser.add_argument("-n", "--intra_community_density", type=float, default=0.1, help="density of intra-community connections")
parser.add_argument("-o", "--inter_community_density", type=float, default=0.01, help="density of inter-community connections")
parser.add_argument("-p", "--reservoir_weight", type=float, default=5000.0, help="reservoir connection weighting factor")
parser.add_argument("-q", "--const_input_weight", type=float, default=5000.0, help="anatomy input weighting factor")
parser.add_argument("-r", "--prediction_input_weight", type=float, default=5000.0, help="previous prediction as input weighting factor")
parser.add_argument("-s", "--device_name", type=str, default="cpu", help="device name")
args = parser.parse_args()

data_directory = args.data_directory
subject_list_file = args.subject_list_file
max_num_subjects = args.max_num_subjects
simulated_time_series_directory_path = args.simulated_time_series_directory_path
training_log_file_path = args.training_log_file_path
sim_steps_before_training = args.sim_steps_before_training
training_reps_per_time_series = args.training_reps_per_time_series
rls_steps_per_data_step = args.rls_steps_per_data_step
sim_steps_per_rls_step = args.sim_steps_per_rls_step
training_reps_per_time_series = args.training_reps_per_time_series
num_communities_int = args.num_communities
neurons_per_community_int = args.neurons_per_community
const_inputs_per_community = args.const_inputs_per_community
reservoir_density = args.reservoir_density
intra_community_density = args.intra_community_density
inter_community_density = args.inter_community_density
reservoir_weight = args.reservoir_weight
const_input_weight = args.const_input_weight
prediction_input_weight = args.prediction_input_weight
device_name = args.device_name

sim_steps_per_data_step = sim_steps_per_rls_step * rls_steps_per_data_step
device = torch.device(device_name)
dtype = torch.float
reservoir_density_tensor = torch.full( (num_communities_int, num_communities_int), inter_community_density, dtype=dtype, device=device ) + (intra_community_density - inter_community_density) * torch.eye( num_communities_int, dtype=dtype, device=device )

# training_subjects = hcp.load_training_subjects(data_directory)
training_subjects = hcp.load_subject_list(subject_list_file)

print_every_seconds = 60
last_print_time = code_start_time
num_subjects = 0
print('starting training')
with open(training_log_file_path, 'w', encoding='UTF8') as training_log_file:
    training_log_writer = csv.writer(training_log_file)
    training_log_writer.writerow(['subject_id', 'time_series', 'rep_type', 'rep', 'time', 'rmse', 'rmse_fc'])
    for subject_id in training_subjects:
        const_input_file_path = hcp.get_area_features_file_path(data_directory, subject_id)
        const_input = hcp.load_matrix_from_binary(const_input_file_path, device=device)
        print(f'Loaded const input data from {const_input_file_path} with size {const_input.size()}')
        # For every subject, make 4 models, each one trained on a different combination of 3 time series and tested on the 4th.
        for left_out_time_series in hcp.time_series_suffixes:
            all_to_all_model = IzhikevichStructuredReservoirComputer( \
                num_communities_int=num_communities_int, \
                neurons_per_community_int=neurons_per_community_int, \
                const_inputs_per_community=const_inputs_per_community, \
                reservoir_density_tensor=reservoir_density_tensor, \
                G=reservoir_weight, \
                Q_const=const_input_weight, \
                Q_prediction=prediction_input_weight, \
                dtype=dtype, \
                device=device \
            )
            model_param_string = all_to_all_model.get_params_string()
            all_to_all_model.set_const_input(const_input)
            for time_series in hcp.time_series_suffixes:
                if time_series != left_out_time_series:
                    data_time_series_file_path = hcp.get_time_series_file_path(data_directory, subject_id, time_series)
                    data_ts = hcp.load_time_series_from_binary(data_time_series_file_path, device=device, normalize='std-mean')
                    print(f'Loaded data time series from {data_time_series_file_path} with size {data_ts.size()}')
                    num_data_time_points = data_ts.size(1)
                    data_fc = torch.corrcoef(data_ts[:,1:])
                    data_ts_init = data_ts[:,0]
                    sim_ts = torch.zeros_like(data_ts)
                    sim_ts[:,0] = data_ts_init
                    for rep_index in range(training_reps_per_time_series):
                        all_to_all_model.reset()
                        for sim_step in range(sim_steps_before_training):
                            all_to_all_model.sim_step()
                        all_to_all_model.prediction = data_ts_init
                        for data_step in range(1,num_data_time_points):
                            # data_ts_at_start = data_ts[:,data_step-1]
                            # data_ts_at_end = data_ts[:,data_step]
                            for rls_step in range(rls_steps_per_data_step):
                                # Linearly interpolate between steps.
                                # data_ts_now = (rls_steps_per_data_step-rls_step-1)/rls_steps_per_data_step*data_ts_at_start + (rls_step+1)/rls_steps_per_data_step*data_ts_at_end
                                # all_to_all_model.sim_steps_then_rls_step_with_const_correct(data_ts_at_start, sim_steps_per_rls_step)
                                for sim_step in range(sim_steps_per_rls_step):
                                    all_to_all_model.sim_step()
                                all_to_all_model.rls_step(data_ts[:,data_step])
                        all_to_all_model.reset()
                        for sim_step in range(sim_steps_before_training):
                            all_to_all_model.sim_step()
                        all_to_all_model.prediction = data_ts_init
                        for data_step in range(1,num_data_time_points):
                            for sim_step in range(sim_steps_per_data_step):
                                all_to_all_model.sim_step()
                            sim_ts[:,data_step] = all_to_all_model.prediction
                        rmse = torch.sqrt(  torch.mean( torch.square(sim_ts[:,1:] - data_ts[:,1:]) )  ).item()
                        sim_fc = torch.corrcoef(sim_ts[:,1:])
                        rmse_fc = torch.sqrt(  torch.mean( torch.square(sim_fc - data_fc) )  ).item()
                        time_since_start = time.time() - code_start_time
                        training_log_writer.writerow([subject_id, time_series, 'training', rep_index, time_since_start, rmse, rmse_fc])
                        print(f'time {time_since_start:.1f} seconds, time series {subject_id} {time_series}, training rep {rep_index}, RMSE {rmse:.3f}, FC RMSE {rmse_fc:.3f}')
            # Now do a single validation rep on the left-out time series.
            time_series = left_out_time_series
            data_time_series_file_path = hcp.get_time_series_file_path(data_directory, subject_id, time_series)
            data_ts = hcp.load_time_series_from_binary(data_time_series_file_path, device=device, normalize='std-mean')
            print(f'Loaded data time series from {data_time_series_file_path} with size {data_ts.size()}')
            num_data_time_points = data_ts.size(1)
            data_fc = torch.corrcoef(data_ts[:,1:])
            data_ts_init = data_ts[:,0]
            sim_ts = torch.zeros_like(data_ts)
            sim_ts[:,0] = data_ts_init
            rep_inde = 0
            all_to_all_model.reset()
            for sim_step in range(sim_steps_before_training):
                all_to_all_model.sim_step()
            all_to_all_model.prediction = data_ts_init
            for data_step in range(1,num_data_time_points):
                sim_ts[:,data_step] = all_to_all_model.sim_steps_then_mean_prediction(sim_steps_per_data_step)
                current_time = time.time()
            rmse = torch.sqrt(  torch.mean( torch.square(sim_ts[:,1:] - data_ts[:,1:]) )  ).item()
            sim_fc = torch.corrcoef(sim_ts[:,1:])
            rmse_fc = torch.sqrt(  torch.mean( torch.square(sim_fc - data_fc) )  ).item()
            time_since_start = time.time() - code_start_time
            training_log_writer.writerow([subject_id, time_series, 'validation', rep_index, time_since_start, rmse, rmse_fc])
            print('saving simulated time series...')
            # See https://pytorch.org/docs/stable/generated/torch.save.html
            torch.save( sim_ts, simulated_time_series_directory_path + f"sim_ts_3vs1_validation_{subject_id}_{time_series}_rperts_{training_reps_per_time_series}_simb4tr_{sim_steps_before_training}_rlsperdt_{rls_steps_per_data_step}_simperrls_{sim_steps_per_rls_step}_{model_param_string}.pt" )
            print(f'time {time_since_start:.1f} seconds, time series {subject_id} {time_series}, validation rep, RMSE {rmse:.3f}, FC RMSE {rmse_fc:.3f}')
            del all_to_all_model
        num_subjects += 1

print(f'done, time {time.time() - code_start_time:.1f}')