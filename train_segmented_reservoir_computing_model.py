# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:21:22 2023

@author: AGCRAIG
"""

import argparse
import torch
import numpy as np
import time as time
import pickle
import csv
from reservoirmltorch import IzhikevichReservoirComputerSegmented
 
code_start_time = time.time()
print('started', __file__)

# Adapted from https://docs.python.org/3/howto/argparse.html#id1
parser = argparse.ArgumentParser(description="Train a segmented reservoir computing model on a series of time series inputs with corresponding constant inputs.")
parser.add_argument("-i", "--training_data_file_pair_list_file_path", type=str, default="training_data_file_pairs.txt", help="list of time series and constant feature file path pairs") 
parser.add_argument("-o", "--model_pickle_file_path", type=str, default="segmented_reservoir_computing_model.bin", help="path to pickle dump file to which to save the trained model")
parser.add_argument("-l", "--training_log_file_path", type=str, default="training_log.csv", help="path to which to save the training log CSV file")
parser.add_argument("-s", "--num_segments", type=int, default=360, help="number of segments")
parser.add_argument("-n", "--neurons_per_segment", type=int, default=1000, help="neurons per segment")
parser.add_argument("-c", "--const_inputs_per_segment", type=int, default=4, help="number of constant inputs per segment")
parser.add_argument("-r", "--reservoir_density", type=float, default=0.1, help="reservoir connection density")
parser.add_argument("-g", "--intra_reservoir_weight", type=float, default=5000.0, help="intra-reservoir connection weighting factor")
parser.add_argument("-w", "--inter_reservoir_weight", type=float, default=5000.0, help="inter-reservoir connection weighting factor")
parser.add_argument("-q", "--const_input_weight", type=float, default=4000.0, help="anatomy input weighting factor")
parser.add_argument("-p", "--prediction_input_weight", type=float, default=4000.0, help="previous prediction as input weighting factor")
parser.add_argument("-d", "--device_name", type=str, default="cpu", help="device name")   
parser.add_argument("-t", "--training_reps_per_time_series", type=int, default=1, help="number of training reps per time series")
parser.add_argument("-a", "--rls_steps_per_data_step", type=int, default=1, help="recursive least squares steps per data time point") 
parser.add_argument("-b", "--sim_steps_per_rls_step", type=int, default=10, help="sim steps per RLS step") 
args = parser.parse_args()

training_data_file_pair_list_file_path = args.training_data_file_pair_list_file_path
model_pickle_file_path = args.model_pickle_file_path
training_log_file_path = args.training_log_file_path
num_segments = args.num_segments
neurons_per_segment = args.neurons_per_segment
const_inputs_per_segment = args.const_inputs_per_segment
reservoir_density = args.reservoir_density
intra_reservoir_weight = args.intra_reservoir_weight
inter_reservoir_weight = args.inter_reservoir_weight
const_input_weight = args.const_input_weight
prediction_input_weight = args.prediction_input_weight
device_name = args.device_name
training_reps_per_time_series = args.training_reps_per_time_series
rls_steps_per_data_step = args.rls_steps_per_data_step
sim_steps_per_rls_step = args.sim_steps_per_rls_step

device = torch.device(device_name)

all_to_all_model = IzhikevichReservoirComputerSegmented( \
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

sim_steps_per_data_step = sim_steps_per_rls_step * rls_steps_per_data_step
print_every_seconds = 60
last_print_time = code_start_time
print('starting training')
ts_index = 0
with open(training_log_file_path, 'w', encoding='UTF8') as training_log_file:
    training_log_writer = csv.writer(training_log_file)
    training_log_writer.writerow(['data_time_series_file_path', 'const_input_file_path', 'rep', 'time', 'rmse', 'rmse_fc'])
    with open(training_data_file_pair_list_file_path, 'r') as training_data_file_pair_list_file:
        for training_data_file_pair_line in training_data_file_pair_list_file:
            training_data_file_pair = training_data_file_pair_line.split(',')
            if len(training_data_file_pair) < 2:
                break

            data_time_series_file_path = training_data_file_pair[0].strip()
            ts_data = torch.from_numpy( np.fromfile(data_time_series_file_path, np.float64) ).to(device)
            (ts_std, ts_mean) = torch.std_mean(ts_data)
            ts_data_normed = (ts_data - ts_mean)/ts_std
            num_data_time_points = len(ts_data_normed)//num_segments
            data_ts = torch.reshape( ts_data_normed, (num_data_time_points, num_segments) ).transpose(0, 1)
            data_fc = torch.corrcoef(data_ts[:,1:])
            data_ts_init = data_ts[:,0]
            sim_ts = torch.zeros_like(data_ts)
            sim_ts[:,0] = data_ts_init
            print(f'Loaded data time series from {data_time_series_file_path} with size {data_ts.size()}')

            const_input_file_path = training_data_file_pair[1].strip()
            const_data = torch.from_numpy( np.fromfile(const_input_file_path, np.float64) ).to(device)
            const_input = torch.reshape( const_data, (const_inputs_per_segment, num_segments) ).transpose(0, 1)
            all_to_all_model.set_const_input(const_input)
            print(f'Loaded data time series from {const_input_file_path} with size {const_input.size()}')

            for rep_index in range(training_reps_per_time_series):

                all_to_all_model.reset()
                all_to_all_model.prediction = data_ts_init
                for data_step in range(1,num_data_time_points):
                    data_ts_now = data_ts[:,data_step]
                    for rls_step in range(rls_steps_per_data_step):
                        all_to_all_model.sim_steps_then_rls_step_with_const_correct(data_ts_now, sim_steps_per_rls_step)
                        current_time = time.time()
                        if current_time - last_print_time >= print_every_seconds:
                            print(f'time {current_time - code_start_time:.1f} seconds, time series {ts_index}, rep {rep_index}, training data step {data_step}, RLS step {rls_step}')
                            last_print_time = current_time
                
                all_to_all_model.reset()
                all_to_all_model.prediction = data_ts_init
                for data_step in range(1,num_data_time_points):
                    sim_ts[:,data_step] = all_to_all_model.sim_steps_then_mean_prediction(sim_steps_per_data_step)
                    current_time = time.time()
                    if current_time - last_print_time >= print_every_seconds:
                        print(f'time {current_time - code_start_time:.1f} seconds, time series {ts_index}, rep {rep_index}, validation data step {data_step}')
                        last_print_time = current_time
                rmse = torch.sqrt(  torch.mean( torch.square(sim_ts[:,1:] - data_ts[:,1:]) )  ).item()
                sim_fc = torch.corrcoef(sim_ts[:,1:])
                rmse_fc = torch.sqrt(  torch.mean( torch.square(sim_fc - data_fc) )  ).item()
                time_since_start = time.time() - code_start_time
                training_log_writer.writerow([data_time_series_file_path, const_input_file_path, rep_index, time_since_start, rmse, rmse_fc])
                print(f'time {time_since_start:.1f} seconds, time series {ts_index}, rep {rep_index}, RMSE {rmse:.3f}, FC RMSE {rmse_fc:.3f}')
                ts_index += 1

print('saving model...')
model_file = open(model_pickle_file_path, 'wb')
pickle.dump(all_to_all_model, model_file)
model_file.close()

print(f'done, time {time.time() - code_start_time:.1f}')