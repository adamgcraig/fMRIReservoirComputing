# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:21:22 2023

@author: AGCRAIG
"""

import argparse
import torch
from torch.utils.data import DataLoader
import time as time
import os
import hcpdatautils as hcp
from reservoirmltorch import IzhikevichReservoirComputerSegmented
 
code_start_time = time.time()
# print('started', __file__)

# Adapted from https://docs.python.org/3/howto/argparse.html#id1
parser = argparse.ArgumentParser(description="Train a segmented reservoir computing model on a series of time series inputs with corresponding constant inputs.")
parser.add_argument("-a", "--data_directory", type=str, default="data", help="base directory under which to find fMRI_ts_binaries and anatomy_binaries")
parser.add_argument("-b", "--trained_model_directory", type=str, default="trained_models", help="directory in which to save pickle dump of trained model")
parser.add_argument("-c", "--device_name", type=str, default="cuda", help="device name")   
parser.add_argument("-d", "--normalization_mode", type=str, default="std-mean", help="method by which to normalize the data, either std-mean, min-max, or none")
parser.add_argument("-e", "--neurons_per_segment", type=int, default=1000, help="neurons per segment")
parser.add_argument("-f", "--num_epochs", type=int, default=1, help="number of times to iterate over the training data set")
parser.add_argument("-g", "--sim_steps_before_training", type=int, default=0, help="sim steps to run before training starts") 
parser.add_argument("-i", "--rls_steps_per_data_step", type=int, default=1, help="recursive least squares steps per data time point") 
parser.add_argument("-j", "--sim_steps_per_rls_step", type=int, default=1, help="sim steps per RLS step")  
parser.add_argument("-k", "--save_every_epochs", type=int, default=20, help="save once every this many epochs")  
parser.add_argument("-l", "--reservoir_density", type=float, default=0.1, help="reservoir connection density")
parser.add_argument("-m", "--intra_reservoir_weight", type=float, default=5000.0, help="intra-reservoir connection weighting factor")
parser.add_argument("-n", "--inter_reservoir_weight", type=float, default=5000.0, help="inter-reservoir connection weighting factor")
parser.add_argument("-o", "--const_input_weight", type=float, default=5000.0, help="anatomy input weighting factor")
parser.add_argument("-p", "--prediction_input_weight", type=float, default=5000.0, help="previous prediction as input weighting factor")
parser.add_argument("-q", "--dt", type=float, default=0.04, help="Euler integration step size")
args = parser.parse_args()

data_directory = args.data_directory
trained_model_directory = args.trained_model_directory
neurons_per_segment = args.neurons_per_segment
reservoir_density = args.reservoir_density
intra_reservoir_weight = args.intra_reservoir_weight
inter_reservoir_weight = args.inter_reservoir_weight
const_input_weight = args.const_input_weight
prediction_input_weight = args.prediction_input_weight
device_name = args.device_name
rls_steps_per_data_step = args.rls_steps_per_data_step
sim_steps_per_rls_step = args.sim_steps_per_rls_step
sim_steps_before_training = args.sim_steps_before_training
save_every_epochs = args.save_every_epochs
normalization_mode = args.normalization_mode
dt = args.dt
num_epochs = args.num_epochs

sim_steps_per_data_step = sim_steps_per_rls_step * rls_steps_per_data_step

output_file_name_string = f"multi_subject_pre_train_{sim_steps_before_training}_rls_to_data_{rls_steps_per_data_step}_sim_to_rls_{sim_steps_per_rls_step}_nps_{neurons_per_segment}_Q_const_{const_input_weight}_Q_pred_{prediction_input_weight}_G_intra_{intra_reservoir_weight}_G_inter_{inter_reservoir_weight}_dt_{dt}_norm_{normalization_mode}.csv"

device = torch.device(device_name)
dtype = torch.float

# print('initialized model...')
model = IzhikevichReservoirComputerSegmented( \
    num_segments=hcp.num_brain_areas, \
    const_inputs_per_segment=hcp.features_per_area, \
    neurons_per_segment=neurons_per_segment, \
    reservoir_density=reservoir_density, \
    G_intra=intra_reservoir_weight, \
    G_inter=inter_reservoir_weight, \
    Q_const=const_input_weight, \
    Q_prediction=prediction_input_weight, \
    dt=dt, \
    device=device \
)

training_dataset = hcp.AnatomyAndTimeSeriesDataset(root_dir=data_directory, dtype=dtype, device=device, normalize=normalization_mode)
validation_dataset = hcp.AnatomyAndTimeSeriesDataset(root_dir=data_directory, validate=True, dtype=dtype, device=device, normalize=normalization_mode)
training_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=1)

sim_ts_step = torch.zeros( (hcp.num_brain_areas,1), dtype=dtype, device=device )
with torch.no_grad():
    print(f'time, epoch, phase, batch, ts, TSRMSE, PSRMSE, FCRMSE')
    training_start_time = time.time()
    for epoch in range(num_epochs):
        # training_size = len(training_dataloader.dataset)
        for batch, (anat_batch, ts_batch) in enumerate(training_dataloader):
            batch_size = anat_batch.size(0)
            for ts_index in range(batch_size):
                anat = anat_batch[ts_index,:,:]
                data_ts = ts_batch[ts_index,:,:]
                data_ps = hcp.get_ps(data_ts)
                data_fc = hcp.get_fc(data_ts)
                model.set_const_input(anat)
                model.train_once(data_ts, sim_steps_before_training=sim_steps_before_training, rls_steps_per_data_step=rls_steps_per_data_step, sim_steps_per_rls_step=sim_steps_per_rls_step)
                sim_ts = model.predict( ts_init=data_ts[:,0], num_data_time_points=data_ts.size(1), sim_steps_before_start=sim_steps_before_training, sim_steps_per_data_step=sim_steps_per_data_step )
                sim_ps = hcp.get_ps(sim_ts)
                sim_fc = hcp.get_fc(sim_ts)
                ts_rmse = hcp.get_ts_rmse_torch(data_ts, sim_ts)
                ps_rmse = hcp.get_ps_rmse_torch(data_ps, sim_ps)
                fc_rmse = hcp.get_fc_rmse_torch(data_fc, sim_fc)
                print( time.time() - training_start_time, epoch, 'training', batch, ts_index, ts_rmse, ps_rmse, fc_rmse )
        for batch, (anat_batch, ts_batch) in enumerate(validation_dataloader):
            batch_size = anat_batch.size(0)
            for ts_index in range(batch_size):
                anat = anat_batch[ts_index,:,:]
                data_ts = ts_batch[ts_index,:,:]
                data_ps = hcp.get_ps(data_ts)
                data_fc = hcp.get_fc(data_ts)
                model.set_const_input(anat)
                sim_ts = model.predict( ts_init=data_ts[:,0], num_data_time_points=data_ts.size(1), sim_steps_before_start=sim_steps_before_training, sim_steps_per_data_step=sim_steps_per_data_step )
                sim_ps = hcp.get_ps(sim_ts)
                sim_fc = hcp.get_fc(sim_ts)
                ts_rmse = hcp.get_ts_rmse_torch(data_ts, sim_ts)
                ps_rmse = hcp.get_ps_rmse_torch(data_ps, sim_ps)
                fc_rmse = hcp.get_fc_rmse_torch(data_fc, sim_fc)
                ts_rmse = hcp.get_ts_rmse_torch(data_ts, sim_ts)
                print( time.time() - training_start_time, epoch, 'validation', batch, ts_index, ts_rmse, ps_rmse, fc_rmse )
        # Save the model at the end of the run.
        if epoch % save_every_epochs == 0:
            trained_model_file_path = os.path.join( trained_model_directory, f"model_{output_file_name_string}_epochs_{num_epochs}.bin")
            model_file = open(trained_model_file_path, 'wb')
            torch.save(model, model_file)
            model_file.close()