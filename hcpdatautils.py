# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:21:22 2023

@author: AGCRAIG
"""
# utilities for loading our human connectome project binary data files
# We want to simplify things but do not want to break working code that uses the original hcpdatautils.py.
# In particular, we want to stick with a single convention for the order of dimensions.
# batch x time x ROI for fMRI time series
# batch x feature x ROI for structural features
# batch x ROI x ROI for structural connectivity
# This works better with torch.nn.Linear layers, where we usually want to act on each row of ROIs independently
# and with the RNN and transformer subclasses, which use this ordering if we set batch_first=True.

import os
import numpy as np
import pandas
import torch
from torch.utils.data import Dataset

# There are always 360 brain areas and 4 features per area.
# There are usually 1200 time points in a data time series.
# There are usually 4 time series per subject.
# Some are missing 2_LR and 2_RL.
# For the "get_..._file_path()" functions,
# we rely on the consistent naming convention we set up
# and assume directory_path ends in the correct path separator character, '/' or '\'.
num_brain_areas = 360
num_time_points = 1200
delta_t = 0.72# seconds between time points
feature_names = ['thickness', 'myelination', 'curvature', 'sulcus depth']
features_per_area = len(feature_names)
time_series_suffixes = ['1_LR', '1_RL', '2_LR', '2_RL']
time_series_per_subject = len(time_series_suffixes)

def load_roi_info(directory_path:str, dtype=torch.float, device='cpu'):
    roi_info = pandas.read_csv( os.path.join(directory_path, 'roi_info.csv') )
    names = roi_info['name'].values
    coords = torch.tensor( data=roi_info[['x','y','z']].values, dtype=dtype, device=device )
    return names, coords

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

# Load the files in the more typical Python way, where the index of the last (column) dimension changes fastest.
# We store data matrices as sequences of 64-bit floating point numbers.
# Specifically, each consists of some number of 360-number blocks where consecutive elements are for different ROIs.
# As such, we need to specify the number of columns as 360 in order to convert back to a 2D matrix.
# If device is None, return a numpy array.
# Otherwise, return a PyTorch tensor.
def load_matrix_from_binary(file_path:str, dtype=torch.float, device='cpu', num_cols:int=num_brain_areas):
    data_matrix = np.fromfile(file_path, np.float64).reshape( (-1, num_cols), order='C' )
    return torch.from_numpy(data_matrix).to(device, dtype=dtype)


def normalize_time_series_torch(time_series, normalize:str=None):
        if normalize == None:
            zero_point = 0.0
            scale_denominator = 1.0
        elif normalize == 'std-mean':
            (scale_denominator, zero_point) = torch.std_mean(time_series)
        elif normalize == 'min-max':
            min_val = torch.min(time_series)
            max_val = torch.max(time_series)
            zero_point = (min_val + max_val)/2
            scale_denominator = (max_val - min_val)/2
        elif normalize == 'min-max-pos':
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

def load_all_time_series_for_subject(directory_path:str, subject_id:int, dtype=torch.float, device='cpu', num_cols:int=num_brain_areas):
    time_series = torch.zeros( (time_series_per_subject, num_time_points, num_cols), dtype=dtype, device=device )
    for ts_index in range(time_series_per_subject):
        ts_suffix = time_series_suffixes[ts_index]
        file_path = get_time_series_file_path(directory_path=directory_path, subject_id=subject_id, time_series_suffix=ts_suffix)
        ts = load_matrix_from_binary(file_path=file_path, dtype=dtype, device=device, num_cols=num_cols)
        # Some time series have fewer than the standard number of time points, but none have more.
        # All of the ones in our selected training, validation, and testing sets have the standard number.
        actual_num_time_points = ts.size(dim=0)
        time_series[ts_index,:actual_num_time_points,:] = ts
    return time_series

def load_all_time_series_for_subjects(directory_path:str, subject_ids:torch.Tensor, dtype=torch.float, device='cpu', num_cols:int=num_brain_areas):
    num_subjects = len(subject_ids)
    time_series = torch.zeros( (num_subjects, time_series_per_subject, num_time_points, num_cols), dtype=dtype, device=device )
    for subject_index in range(num_subjects):
        time_series[subject_index,:,:,:] = load_all_time_series_for_subject(directory_path=directory_path, subject_id=subject_ids[subject_index], dtype=dtype, device=device, num_cols=num_cols)
    return time_series

def get_fc(ts:torch.Tensor):
    # We want to take the correlations between pairs of areas over time.
    # torch.corrcoef() assumes observations are in columns and variables in rows.
    # As such, we have to take the transpose.
    return torch.corrcoef( ts.transpose(dim0=0, dim1=1) )

def get_fc_batch(ts_batch:torch.Tensor):
    # Take the FC of each individual time series.
    batch_size = ts_batch.size(dim=0)
    num_rois = ts_batch.size(dim=-1)
    fc = torch.zeros( (batch_size, num_rois, num_rois), dtype=ts_batch.dtype, device=ts_batch.device )
    for ts_index in range(batch_size):
        fc[ts_index,:,:] = get_fc(ts_batch[ts_index,:,:])
    return fc

def get_fcd(ts:torch.Tensor, window_length:int=90, window_step:int=1):
    # Calculate functional connectivity dynamics.
    # Based on Melozzi, Francesca, et al. "Individual structural features constrain the mouse functional connectome." Proceedings of the National Academy of Sciences 116.52 (2019): 26961-26969.
    dtype = ts.dtype
    device = ts.device
    ts_size = ts.size()
    T = ts_size[0]
    R = ts_size[1]
    triu_index = torch.triu_indices(row=R, col=R, offset=1, dtype=torch.int, device=device)
    triu_row = triu_index[0]
    triu_col = triu_index[1]
    num_indices = triu_index.size(dim=1)
    left_window_margin = window_length//2 + 1
    right_window_margin = window_length - left_window_margin
    window_centers = torch.arange(start=left_window_margin, end=T-left_window_margin+1, step=window_step, dtype=torch.int, device=device)
    window_offsets = torch.arange(start=-right_window_margin, end=left_window_margin, dtype=torch.int, device=device)
    num_windows = window_centers.size(dim=0)
    window_fc = torch.zeros( (num_windows, num_indices), dtype=dtype, device=device )
    for c in range(num_windows):
        fc = get_fc(ts[window_offsets+window_centers[c],:])
        window_fc[c,:] = fc[triu_row, triu_col]
    return torch.corrcoef(window_fc)

def get_ps(ts:torch.Tensor, time_resolution:torch.float=delta_t):
    time_dim = 0
    ts_fft_abs_sq = torch.fft.rfft(ts,dim=time_dim).abs().square()
    num_halves = torch.full_like(input=ts_fft_abs_sq, fill_value=2.0)
    num_halves[0,:] = 1.0
    num_halves[-1,:] = 1.0
    num_time_points = ts.size(dim=time_dim)
    return ts_fft_abs_sq * num_halves * (time_resolution/num_time_points)

def get_ps_batch(ts_batch:torch.Tensor, time_resolution:torch.float=delta_t):
    time_dim = 1
    ts_fft_abs_sq = torch.fft.rfft(ts_batch,dim=time_dim).abs().square()
    num_halves = torch.full_like(input=ts_fft_abs_sq, fill_value=2.0)
    num_halves[:,0,:] = 1.0
    num_halves[:,-1,:] = 1.0
    num_time_points = ts_batch.size(dim=time_dim)
    return ts_fft_abs_sq * num_halves * (time_resolution/num_time_points)

# only assumes the tensors are the same shape
def get_rmse(tensor1:torch.Tensor, tensor2:torch.Tensor):
    return torch.sqrt(  torch.mean( torch.square(tensor2 - tensor1) )  ).item()

# takes the mean over all but the first dimension
# For our 3D batch-first tensors, this gives us a 1D tensor of RMSE values, one for each batch.
def get_rmse_batch(tensor1:torch.Tensor, tensor2:torch.Tensor):
    dim_indices = tuple(   range(  len( tensor1.size() )  )   )
    return torch.sqrt(  torch.mean( torch.square(tensor2 - tensor1), dim=dim_indices[1:] )  )

# In several cases, we have symmetric square matrices with fixed values on the diagonal.
# In particular, this is true of the functional connectivity and structural connectivity matrices.
# For such matrices, it is more meaningful to only calculate the RMSE of the elements above the diagonal.
def get_triu_rmse(tensor1:torch.Tensor, tensor2:torch.Tensor):
    indices = torch.triu_indices( row=tensor1.size(0), col=tensor1.size(1), offset=1, device=tensor1.device )
    indices_r = indices[0]
    indices_c = indices[1]
    return get_rmse( tensor1[indices_r,indices_c], tensor2[indices_r,indices_c] )

def get_triu_rmse_batch(tensor1:torch.Tensor, tensor2:torch.Tensor):
    indices = torch.triu_indices( tensor1.size(-2), tensor1.size(-1), 1 )
    indices_r = indices[0]
    indices_c = indices[1]
    batch_size = tensor1.size(dim=0)
    num_triu_elements = indices.size(dim=1)
    dtype = tensor1.dtype
    device = tensor1.device
    triu1 = torch.zeros( (batch_size, num_triu_elements), dtype=dtype, device=device )
    triu2 = torch.zeros_like(triu1)
    for b in range(batch_size):
        triu1[b,:] = tensor1[b,indices_r,indices_c]
        triu2[b,:] = tensor2[b,indices_r,indices_c]
    return get_rmse_batch(triu1, triu2)

def get_triu_corr(tensor1:torch.Tensor, tensor2:torch.Tensor):
    indices = torch.triu_indices( row=tensor1.size(0), col=tensor1.size(1), offset=1, device=tensor1.device )
    indices_r = indices[0]
    indices_c = indices[1]
    tensor1_triu = tensor1[indices_r,indices_c].unsqueeze(dim=0)
    # print( tensor1_triu.size() )
    tensor2_triu = tensor2[indices_r,indices_c].unsqueeze(dim=0)
    # print( tensor2_triu.size() )
    tensor_pair_triu = torch.cat( (tensor1_triu,tensor2_triu), dim=0 )
    # print( tensor_pair_triu.size() )
    corr = torch.corrcoef(tensor_pair_triu)
    # print(corr)
    return corr[0,1]

def get_triu_corr_batch(tensor1:torch.Tensor, tensor2:torch.Tensor):
    indices = torch.triu_indices( tensor1.size(-2), tensor1.size(-1), 1 )
    indices_r = indices[0]
    indices_c = indices[1]
    batch_size = tensor1.size(dim=0)
    dtype = tensor1.dtype
    device = tensor1.device
    corr_batch = torch.zeros( (batch_size,), dtype=dtype, device=device )
    for b in range(batch_size):
        triu_pair = torch.stack( (tensor1[b,indices_r,indices_c], tensor2[b,indices_r,indices_c]) )
        pair_corr = torch.corrcoef(triu_pair)
        corr_batch[b] = pair_corr[0,1]
    return corr_batch

# Returns fMRI time series, DT MRI SC, structural MRI features, subject ID, and time series suffix.
class TripleFMRIDataset(Dataset):
    def __init__(self, root_dir:str, subject_set:str='train', max_subjects:int=None, dtype=None, device=None, normalize:str='none', num_cols:int=num_brain_areas):
        self.root_dir = root_dir
        if subject_set == 'test':
            subject_list_map = load_testing_subjects(self.root_dir)
        elif subject_set == 'validate':
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
        self.num_cols = num_cols

    def __len__(self):
        return self.num_time_series
    
    def __getitem__(self, index:int):
        time_series_per_subject = len(time_series_suffixes)
        subject_index = index//time_series_per_subject
        ts_index = index % time_series_per_subject
        subject_id = self.subject_list[subject_index]
        time_series_suffix = time_series_suffixes[ts_index]
        ts = normalize_time_series_torch(  load_matrix_from_binary( get_time_series_file_path(self.root_dir, subject_id, time_series_suffix), dtype=self.dtype, device=self.device, num_cols=self.num_cols ), normalize=self.normalize  )
        sc = load_matrix_from_binary( get_structural_connectivity_file_path(self.root_dir, subject_id), dtype=self.dtype, device=self.device, num_cols=self.num_cols )
        anat = load_matrix_from_binary( get_area_features_file_path(self.root_dir, subject_id), dtype=self.dtype, device=self.device, num_cols=self.num_cols )
        return ts, sc, anat, subject_id, time_series_suffix