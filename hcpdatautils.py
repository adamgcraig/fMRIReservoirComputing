# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:21:22 2023

@author: AGCRAIG
"""
# utilities for loading our human connectome project binary data files

import torch
from torch.utils.data import Dataset
from torch.fft import fft
import numpy as np
import os
# import csv

# There are always 360 brain areas and 4 features per area.
# There are usually 1200 time points in a data time series.
# There are usually 4 time series per subject.
# Some are missing 2_LR and 2_RL.
# For the "get_..._file_path()" functions,
# we rely on the consistent naming convention we set up
# and assume directory_path ends in the correct path separator character, '/' or '\'.
num_brain_areas = 360
features_per_area = 4
default_num_data_time_points = 1200
time_series_suffixes = ['1_LR', '1_RL', '2_LR', '2_RL']

def get_area_features_file_path(directory_path:str, subject_id:int):
    return os.path.join(directory_path, 'anatomy_binaries', f"anatomy_{subject_id}.bin")

def get_area_feature_means_file_path(directory_path:str):
    return os.path.join(directory_path, 'anatomy_binaries', f"anatomy_mean.bin")

def get_area_feature_stds_file_path(directory_path:str):
    return os.path.join(directory_path, 'anatomy_binaries', f"anatomy_std.bin")

def get_structural_connectivity_file_path(directory_path:str, subject_id:int):
    return os.path.join(directory_path, 'dtMRI_binaries', f"sc_{subject_id}.bin")

def get_time_series_file_path(directory_path:str, subject_id:int, time_series_suffix:str):
    return os.path.join(directory_path, 'fMRI_ts_binaries', f"ts_{subject_id}_{time_series_suffix}.bin")

# We store data matrices is 64-bit floating point numbers in column-major order.
# As such, we need to specify the number of rows in order to convert back to a 2D matrix.
# For all of our data, the number of rows is num_brain_areas=360.
# If device is None, return a numpy array.
# Otherwise, return a PyTorch tensor.
def load_matrix_from_binary(file_path:str, dtype=None, device=None, num_rows:int=num_brain_areas):
    data_matrix = np.fromfile(file_path, np.float64).reshape( (num_rows, -1), order='F' )
    if device == None:
        return data_matrix
    else:
        if dtype == None:
            dtype = torch.float
        return torch.from_numpy(data_matrix).to(device, dtype=dtype)

# If device=None, returns a numpy array.
# Otherwise, returns a PyTorch tensor on the specified device.
# If normalize=None, returns the values as loaded from the file.
# If normalize='std-mean', returns values normalized to have 0 mean and unit variance.
# If normalize='min-max', returns values rescaled and shifted so that the range is [-1, 1].
# If normalize='min-max-pos', returns values rescaled and shifted so that the range is [0, 1].
def load_time_series_from_binary(file_path:str, dtype=None, device=None, normalize:str=None, num_rows:int=num_brain_areas):
    time_series = load_matrix_from_binary(file_path, dtype, device, num_rows)
    if normalize != None:
        if normalize == 'std-mean':
            if device == None:
                scale_denominator = np.std(time_series)
                zero_point = np.mean(time_series)
            else:
                (scale_denominator, zero_point) = torch.std_mean(time_series)
        elif normalize == 'min-max':
            if device == None:
                min_val = np.min(time_series)
                max_val = np.max(time_series)
            else:
                min_val = torch.min(time_series)
                max_val = torch.max(time_series)
            zero_point = (min_val + max_val)/2
            scale_denominator = (max_val - min_val)/2
        elif normalize == 'min-max-pos':
            if device == None:
                min_val = np.min(time_series)
                max_val = np.max(time_series)
            else:
                min_val = torch.min(time_series)
                max_val = torch.max(time_series)
            zero_point = min_val
            scale_denominator = max_val - min_val
        elif normalize == 'none':
            zero_point = 0.0
            scale_denominator = 1.0
        else:
            print(f'{normalize} is not a recognized type of normalization. Use std-mean, min-max, or none.')
            zero_point = 0.0
            scale_denominator = 1.0
        return (time_series - zero_point)/scale_denominator
    else:
        return time_series

def load_all_time_series_for_subject(directory_path:str, subject_id:int, dtype=torch.float, device='cpu', normalize:str=None, num_rows:int=num_brain_areas):
    time_series_per_subject = len(time_series_suffixes)
    time_series = torch.zeros( (time_series_per_subject, num_rows, default_num_data_time_points), dtype=dtype, device=device )
    for ts_index in range(time_series_per_subject):
        ts_suffix = time_series_suffixes[ts_index]
        file_path = get_time_series_file_path(directory_path=directory_path, subject_id=subject_id, time_series_suffix=ts_suffix)
        time_series[ts_index,:,:] = load_time_series_from_binary(file_path=file_path, dtype=dtype, device=device, normalize=normalize, num_rows=num_rows)
    return time_series
    
def load_subject_list(file_path:str):
    with open(file_path, 'r', encoding='utf-8') as id_file:
        subject_list = list(  map( int, id_file.read().split() )  )
        return subject_list

def load_training_subjects(directory_path:str):
    return load_subject_list( os.path.join(directory_path, 'training_subject_ids.txt') )

def load_validation_subjects(directory_path:str):
    return load_subject_list( os.path.join(directory_path, 'validation_subject_ids.txt') )

def load_testing_subjects(directory_path:str):
    return load_subject_list( os.path.join(directory_path, 'testing_subject_ids.txt') )

def get_has_sc_subject_list(directory_path:str, subject_list:list):
    return list(  filter(lambda subject_id: os.path.isfile( get_structural_connectivity_file_path(directory_path, subject_id) ), subject_list)  )

def get_fc(ts:torch.Tensor, start_index:int=1):
    return torch.corrcoef(ts[:,start_index:])

def get_ps_one_area(ts:torch.Tensor, sampling_frequency:torch.float):
    num_time_points = ts.size(dim=0)
    num_distinct_freqs = num_time_points//2
    num_halves = torch.full( (num_distinct_freqs,), 2, dtype=ts.dtype, device=ts.device )
    num_halves[0] = 1
    num_halves[-1] = 1
    return num_halves * torch.pow(  torch.abs( fft(ts)[0:num_distinct_freqs] ), 2  ) / (sampling_frequency*num_time_points)

def get_ps(ts:torch.Tensor, sampling_frequency:torch.float=0.72):
    num_dimensions = ts.size(dim=0)
    num_distinct_freqs = ts.size(dim=1)//2
    ps = torch.zeros( (num_dimensions, num_distinct_freqs), dtype=ts.dtype, device=ts.device )
    for d in range(num_dimensions):
        ps[d,:] = get_ps_one_area(ts[d,:], sampling_frequency)
    return ps

# only assumes the tensors are the same shape
def get_rmse_torch(tensor1:torch.Tensor, tensor2:torch.Tensor):
    return torch.sqrt(  torch.mean( torch.square(tensor2 - tensor1) )  ).item()

# assumes the tensors are rectangular with time series dimensions in rows and time points in columns
def get_ts_rmse_torch(ts1:torch.Tensor, ts2:torch.Tensor, start_index:int=1, end_index=-1):
    return get_rmse_torch(ts1[:,start_index:end_index], ts2[:,start_index:end_index])

# assumes the tensors are rectangular with time series dimensions in rows and frequencies in columns
def get_ps_rmse_torch(ps1:torch.Tensor, ps2:torch.Tensor, start_index:int=0, end_index:int=-1):
    return get_rmse_torch(ps1[:,start_index:end_index], ps2[:,start_index:end_index])

# assumes the tensors are square matrices of correlation coefficients for area pairs
# computes RMSE over only the elements above the main diagonal
# The FC matrix is symmetric, and the main diagonal is by definition 1,
# so this is the more statistically meaningful way of calculating the FC.
def get_fc_rmse_torch(fc1:torch.Tensor, fc2:torch.Tensor):
    indices = torch.triu_indices( fc1.size(0), fc1.size(1), 1 )
    return get_rmse_torch( fc1[indices], fc2[indices] )

# This version loads a time series and a corresponding anatomy feature file,
# normalizes the time series if so desired,
# and returns
# a num_brain_areas x (features_per_brain_area+1) X matrix
# where the first column is the first time point in the time series
# and the remaining columns are the anatomical features,
# a num_brain_areas x (num_data_time_points-1) Y matrix
# consisting of all the time points after the first
class InitialStateAndAnatomyToTimeSeriesDataset(Dataset):
    def __init__(self, root_dir:str, validate:bool=False, test:bool=False, max_subjects:int=None, dtype=None, device=None, normalize:str=None, num_rows:int=num_brain_areas):
        self.root_dir = root_dir
        if test:
            subject_list_map = load_testing_subjects(self.root_dir)
        elif validate:
            subject_list_map = load_validation_subjects(self.root_dir)
        else:
            subject_list_map = load_training_subjects(self.root_dir)
        self.subject_list = list(subject_list_map)
        num_subjects = len(self.subject_list)
        if max_subjects:
            num_subjects = min( num_subjects, max_subjects )
        self.num_time_series = num_subjects * len(time_series_suffixes)
        self.dtype = dtype
        self.device = device
        self.normalize = normalize
        self.num_rows = num_rows

    def __len__(self):
        return self.num_time_series
    
    def __getitem__(self, index:int):
        time_series_per_subject = len(time_series_suffixes)
        subject_index = index//time_series_per_subject
        ts_index = index % time_series_per_subject
        subject_id = self.subject_list[subject_index]
        time_series_suffix = time_series_suffixes[ts_index]
        ts_all = load_time_series_from_binary( get_time_series_file_path(self.root_dir, subject_id, time_series_suffix), dtype=self.dtype, device=self.device, normalize=self.normalize, num_rows=self.num_rows )
        anat = load_matrix_from_binary( get_area_features_file_path(self.root_dir, subject_id), dtype=self.dtype, device=self.device, num_rows=self.num_rows )
        init_anat = torch.cat(  (  ts_all[:,0].unsqueeze(1), anat ), dim=1  )
        ts_after = ts_all[:,1:]
        return init_anat, ts_after

# Returns the anatomy features first, the time series second.
class AnatomyAndTimeSeriesDataset(Dataset):
    def __init__(self, root_dir:str, validate:bool=False, test:bool=False, max_subjects:int=None, dtype=None, device=None, normalize:str=None, num_rows:int=num_brain_areas):
        self.root_dir = root_dir
        if test:
            subject_list_map = load_testing_subjects(self.root_dir)
        elif validate:
            subject_list_map = load_validation_subjects(self.root_dir)
        else:
            subject_list_map = load_training_subjects(self.root_dir)
        self.subject_list = list(subject_list_map)
        num_subjects = len(self.subject_list)
        if max_subjects:
            num_subjects = min( num_subjects, max_subjects )
        self.num_time_series = num_subjects * len(time_series_suffixes)
        self.dtype = dtype
        self.device = device
        self.normalize = normalize
        self.num_rows = num_rows

    def __len__(self):
        return self.num_time_series
    
    def __getitem__(self, index:int):
        time_series_per_subject = len(time_series_suffixes)
        subject_index = index//time_series_per_subject
        ts_index = index % time_series_per_subject
        subject_id = self.subject_list[subject_index]
        time_series_suffix = time_series_suffixes[ts_index]
        ts = load_time_series_from_binary( get_time_series_file_path(self.root_dir, subject_id, time_series_suffix), dtype=self.dtype, device=self.device, normalize=self.normalize, num_rows=self.num_rows )
        anat = load_matrix_from_binary( get_area_features_file_path(self.root_dir, subject_id), dtype=self.dtype, device=self.device, num_rows=self.num_rows )
        return anat, ts

# Returns the anatomy features first, the time series second.
class AnatomyAndTimeSeriesDatasetV2(Dataset):
    def __init__(self, root_dir:str, validate:bool=False, test:bool=False, max_subjects:int=None, dtype=None, device=None, normalize:str=None, num_rows:int=num_brain_areas):
        self.root_dir = root_dir
        if test:
            subject_list_map = load_testing_subjects(self.root_dir)
        elif validate:
            subject_list_map = load_validation_subjects(self.root_dir)
        else:
            subject_list_map = load_training_subjects(self.root_dir)
        self.subject_list = list(subject_list_map)
        num_subjects = len(self.subject_list)
        if max_subjects:
            num_subjects = min( num_subjects, max_subjects )
        self.num_time_series = num_subjects * len(time_series_suffixes)
        self.dtype = dtype
        self.device = device
        self.normalize = normalize
        self.num_rows = num_rows

    def __len__(self):
        return self.num_time_series
    
    def __getitem__(self, index:int):
        time_series_per_subject = len(time_series_suffixes)
        subject_index = index//time_series_per_subject
        ts_index = index % time_series_per_subject
        subject_id = self.subject_list[subject_index]
        time_series_suffix = time_series_suffixes[ts_index]
        ts = load_time_series_from_binary( get_time_series_file_path(self.root_dir, subject_id, time_series_suffix), dtype=self.dtype, device=self.device, normalize=self.normalize, num_rows=self.num_rows )
        anat = load_matrix_from_binary( get_area_features_file_path(self.root_dir, subject_id), dtype=self.dtype, device=self.device, num_rows=self.num_rows )
        return anat, ts, subject_id, time_series_suffix

# Returns fMRI time series, DT MRI SC, structural MRI features, subject ID, and time series suffix.
class TripleFMRIDataset(Dataset):
    def __init__(self, root_dir:str, validate:bool=False, test:bool=False, max_subjects:int=None, dtype=None, device=None, normalize:str=None, num_rows:int=num_brain_areas):
        self.root_dir = root_dir
        if test:
            subject_list_map = load_testing_subjects(self.root_dir)
        elif validate:
            subject_list_map = load_validation_subjects(self.root_dir)
        else:
            subject_list_map = load_training_subjects(self.root_dir)
        self.subject_list = get_has_sc_subject_list( self.root_dir, list(subject_list_map) )
        num_subjects = len(self.subject_list)
        if max_subjects:
            num_subjects = min( num_subjects, max_subjects )
        self.num_time_series = num_subjects * len(time_series_suffixes)
        self.dtype = dtype
        self.device = device
        self.normalize = normalize
        self.num_rows = num_rows

    def __len__(self):
        return self.num_time_series
    
    def __getitem__(self, index:int):
        time_series_per_subject = len(time_series_suffixes)
        subject_index = index//time_series_per_subject
        ts_index = index % time_series_per_subject
        subject_id = self.subject_list[subject_index]
        time_series_suffix = time_series_suffixes[ts_index]
        ts = load_time_series_from_binary( get_time_series_file_path(self.root_dir, subject_id, time_series_suffix), dtype=self.dtype, device=self.device, normalize=self.normalize, num_rows=self.num_rows )
        sc = load_matrix_from_binary( get_structural_connectivity_file_path(self.root_dir, subject_id), dtype=self.dtype, device=self.device, num_rows=self.num_rows )
        anat = load_matrix_from_binary( get_area_features_file_path(self.root_dir, subject_id), dtype=self.dtype, device=self.device, num_rows=self.num_rows )
        return ts, sc, anat, subject_id, time_series_suffix