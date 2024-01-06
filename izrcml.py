#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:26:00 2023

@author: AGCRAIG
"""
# Izhikevich neuronal network-based reservoir computing machine learning with community structure
# Each community of neurons produces a single output as a weighted sum of the output current values, r, of its individual neurons.
# These neurons then receive this output value as feedback input.

import torch
import numpy as np
import time

class IzhikevichStructuredReservoirComputer:
    # These parameter defaults get overwritten in the constructor,
    # but I am leaving them here for illustrative/explanatory purposes.

    # Izhikevich neuron model parameters
    dt = 0.04# ms, Euler integration step size
    I_bias = 1000.0# pA, constant bias current
    C = 250.0# microF, membrane capacitance
    k = 2.5# ns/mV, scaling factor of action potential half-width
    v_peak = 30.0# mV, peak voltage, maximum v attained during a spike
    v_reset = -65.0# mV, reset potential, v becomes this after a spike
    v_resting = -60.0# mV, resting v
    v_threshold = -40.0# mV, threshold v (when b=0 and I_bias=0)
    a = 0.002# ms^-1, reciprocal of u time constant
    # If we set b to something non-0, go back into the code and swap back in the version of the u and v updates that uses b.
    b = 0.0# nS, sensitivity of u to subthreshold oscillations of v
    d = 100.0# pA, incremental increase in u after each spike
    tau_d = 20.0# ms, synaptic decay time
    tau_r = 2.0# ms, synaptic rise time
    
    # Derived Izhikevich neuron model parameters
    dt_over_C = dt / C
    dt_a = dt * a
    one_minus_dt_a = 1 - dt_a
    dt_a_b = dt_a * b
    exp_neg_dt_over_tau_d = np.exp(-dt / tau_d)
    exp_neg_dt_over_tau_r = np.exp(-dt / tau_r) 
    one_over_tau_r_tau_d = 1 / (tau_r * tau_d)
    
    #  Network properties
    num_communities = 360# number of distinct reservoirs
    neurons_per_community_tensor = 1000# number of neurons in reservoir network for a single community
    # Can be a scalar if communities are all the same size or a 1-D list or tensor of length num_communities if they have different sizes.
    const_inputs_per_community = None# number of constant input values per community
    num_ts_inputs = None# dimensionality of any supplementary time series input
    Q_const = None# weighting factor for constant inputs
    Q_ts = None# weighting factor for time series inputs
    Q_prediction = 400.0# weighting factor for previous prediction fed back in as input
    G = 5000.0# weighting factor of connections between neurons
    # Can be a scalar if pairs of communities all have the same weighting factor or a num_communities x num_communities list of lists or 2-D tensor otherwise.
    reservoir_density=0.1# density of synaptic connections in the reservoir network
    # Can be a scalar if pairs of communities all have the same density or a num_communities x num_communities list of lists or 2-D tensor otherwise.
    P_regularization_constant = 2.0# a constant that weights the correlation matrix P in favor of self-correlation

    # The parameters for which we do not include type hints are the ones that should match the passed-in dtype value,
    # except for dtype itself and device.
    # For acceptable types to use for these, see the PyTorch documentation.
    # In general, dtype should be a floating-point datatype as reflected in the default values for the parameters,
    # but you can specify it to be 32-bit or 64-bit.
    def __init__(self,\
         neurons_per_community_tensor:torch.Tensor=None, \
         neurons_per_community_list:list=None, \
         neurons_per_community_int:int=1000, \
         reservoir_density_tensor:torch.Tensor=None, \
         reservoir_density_list:list=None, \
         reservoir_density_float:float=0.1, \
         num_communities_int:int=360, \
         const_inputs_per_community:int=None,\
         num_ts_inputs:int=None,\
         Q_const=None,\
         Q_ts=None,\
         Q_prediction=400.0,\
         G=5000.0,\
         P_regularization_constant=2.0,\
         dt=0.04,\
         I_bias=1000.0,\
         C=250.0,\
         k=2.5,\
         v_peak=30.0,\
         v_reset=-65.0,\
         v_resting=-60.0,\
         v_threshold=-40.0,\
         a=0.002,\
         b=0.0,\
         d=100.0,\
         tau_d=20.0,\
         tau_r=2.0,\
         device='cpu',
         dtype=torch.float):
        with torch.no_grad():
            init_start_time = time.time()
            print('starting model init...')
            # Store all the scalar constants.
            self.Q_const = Q_const
            self.Q_ts = Q_ts
            self.Q_prediction = Q_prediction
            self.G = G
            self.P_regularization_constant = P_regularization_constant
            self.dt = dt
            self.I_bias = I_bias
            self.C = C
            self.k = k
            self.v_peak = v_peak
            self.v_reset = v_reset
            self.v_resting = v_resting
            self.v_threshold = v_threshold
            self.a = a
            self.b = b
            self.d = d
            self.tau_d = tau_d
            self.tau_r = tau_r
            self.device = device
            self.dtype = dtype
            # Calculate some constants from the passed-in or default values of other constants.
            self.dt_over_C = self.dt / self.C
            self.dt_a = self.dt * self.a
            self.one_minus_dt_a = 1 - self.dt_a
            self.dt_a_b = self.dt_a * self.b
            self.exp_neg_dt_over_tau_d = np.exp(-self.dt / self.tau_d)
            self.exp_neg_dt_over_tau_r = np.exp(-self.dt / self.tau_r) 
            self.one_over_tau_r_tau_d = 1 / (self.tau_r * self.tau_d)
            print(f'set scalar constants {time.time() - init_start_time:.3f}')
            # Figure out which of the options for setting the community sizes the user chose.
            # Convert the arguments used into a tensor of sizes to pass into other methods.
            if neurons_per_community_tensor == None:
                if neurons_per_community_list == None:
                    neurons_per_community_tensor = torch.full( (num_communities_int,), neurons_per_community_int, dtype=torch.int, device=self.device )
                else:
                    neurons_per_community_tensor = torch.tensor( neurons_per_community_list, dtype=torch.int, device=self.device )
            self.neurons_per_community_tensor = neurons_per_community_tensor
            num_neurons = torch.sum(neurons_per_community_tensor)
            num_communities = torch.numel(neurons_per_community_tensor)
            print(f'set neurons per community tensor {time.time() - init_start_time:.3f}')
            if reservoir_density_tensor == None:
                if reservoir_density_list == None:
                    reservoir_density_tensor = torch.full( (num_communities_int,num_communities_int), reservoir_density_float, dtype=torch.float, device=self.device )
                else:
                    reservoir_density_tensor = torch.tensor( reservoir_density_list, dtype=torch.float, device=self.device )
            print(f'set reservoir density tensor {time.time() - init_start_time:.3f}')
            # Randomly generate the fixed network topology.
            # Const and supplementary ts inputs are optional, while prediction inputs and reservoirs must be part of the model.
            # If you want to remove the feedback of the prediction into the model, you can set Q_prediction to 0.
            if const_inputs_per_community == None:
                self.const_input_weights = None
            else:
                self.const_input_weights = __class__.make_const_input_weights(const_inputs_per_community, neurons_per_community_tensor, Q=self.Q_const, dtype=self.dtype, device=self.device)# encoding weights of const input
            print(f'set const input weights {time.time() - init_start_time:.3f}')
            if num_ts_inputs == None:
                self.ts_input_weights = None
            else:
                self.ts_input_weights = __class__.make_ts_input_weights(num_ts_inputs, num_neurons, self.Q_ts, dtype=self.dtype, device=self.device)# encoding weights of ts input
            print(f'set ts input weights {time.time() - init_start_time:.3f}')
            self.prediction_input_weights = __class__.make_prediction_input_weights(neurons_per_community_tensor, self.Q_prediction, dtype=self.dtype, device=self.device)# encoding weights of previous prediction 
            print(f'set prediction input weights {time.time() - init_start_time:.3f}')
            self.reservoir_weights = __class__.make_reservoir_weights(neurons_per_community_tensor, reservoir_density_tensor, self.G, dtype=self.dtype, device=self.device)# synapse weights within the reservoir
            print(f'set reservoir weights {time.time() - init_start_time:.3f}')
            # Initialize the matrices we update during FORCE training.
            self.output_weights = __class__.make_output_weights(neurons_per_community_tensor, dtype=self.dtype, device=self.device)# output weights used to generate prediction from r
            print(f'set output weights {time.time() - init_start_time:.3f}')
            self.P = __class__.make_correlation_matrix(neurons_per_community_tensor, self.P_regularization_constant, dtype=self.dtype, device=self.device) # self.P_regularization_constant * torch.eye(neurons_per_community, neurons_per_community)# "network estimate of the inverse of the correlation matrix" according to the paper
            print(f'set P {time.time() - init_start_time:.3f}')
            # Set the initial neuron state vectors.
            self.state_shape = (num_neurons, 1)
            self.v = self.v_resting + (self.v_peak - self.v_resting) * torch.rand(self.state_shape, dtype=self.dtype, device=self.device)# mV, membrane potential 
            self.state_zeros = torch.zeros(self.state_shape, dtype=self.dtype, device=self.device)
            self.I_synapse = self.state_zeros.clone()# pA, post-synaptic current 
            self.u = self.state_zeros.clone()# pA, adaptation current 
            self.is_spike = self.state_zeros.clone()# 1.0 if the neuron is spiking, 0 otherwise
            self.h = self.state_zeros.clone()# pA/ms, synaptic current gating variable? 
            self.hr = self.state_zeros.clone()# pA/ms, output current gating variable? 
            self.r = self.state_zeros.clone()# pA, network output before transformation by output weights
            self.prediction_shape = (num_communities, 1)
            self.prediction = torch.zeros(self.prediction_shape, dtype=self.dtype, device=self.device)# predicted value of the time series.
            # If no other constant input is set, I_const is just I_bias.
            self.I_const = self.I_bias
            # By default, I_ts is nothing.
            self.I_ts = 0
            print(f'set state vectors {time.time() - init_start_time:.3f}')
            # Store the initial states of the neurons so we can reset to them later.
            self.set_saved_state()
            self.set_initial_output(self.prediction)
            print(f'set saved state {time.time() - init_start_time:.3f}')
    
    # Store the current states of the neurons we can reset to them later.
    # This does not alter the learned values of P or output_weights or the current prediction value.
    def set_saved_state(self):
        with torch.no_grad():
            self.v_0 = self.v.clone()
            self.I_synapse_0 = self.I_synapse.clone()
            self.u_0 = self.u.clone()
            self.is_spike_0 = self.is_spike.clone()
            self.h_0 = self.h.clone()
            self.hr_0 = self.hr.clone()
            self.r_0 = self.r.clone()
    
    def reset_to_saved_state(self):
        with torch.no_grad():
            self.v = self.v_0.clone()
            self.I_synapse = self.I_synapse_0.clone()
            self.u = self.u_0.clone()
            self.is_spike = self.is_spike_0.clone()
            self.h = self.h_0.clone()
            self.hr = self.hr_0.clone()
            self.r = self.r_0.clone()
    
    def set_initial_output(self, ts_init:torch.Tensor):
        with torch.no_grad():
            self.prediction_0 = ts_init.clone()
    
    def reset_output(self):
        with torch.no_grad():
            self.prediction = self.prediction_0.clone()

    def set_const_input(self, const_input: torch.Tensor):
        with torch.no_grad():
            self.I_const = self.I_bias + self.const_input_weights.matmul(  const_input.reshape( (-1,1) )  ).to_dense()
    
    def set_ts_input(self, ts_input: torch.Tensor):
        with torch.no_grad():
            self.I_ts = self.ts_input_weights.matmul(ts_input).to_dense()
    
    # Pass in an arbitrary external current.
    def sim_step_with_external_current(self, I_ext=0):
        with torch.no_grad():
            I_pred = self.prediction_input_weights.matmul(self.prediction)
            I = self.I_synapse + self.I_const + I_ext + I_pred
            # v_minus_v_resting = v - v_resting
            # v = v + dt_over_C .* ( k .* v_minus_v_resting .* (v - v_threshold) - u + I )
            # u = one_minus_dt_a .* u + dt_a_b .* v_minus_v_resting
            self.v += self.dt_over_C * ( self.k * (self.v - self.v_resting) * (self.v - self.v_threshold) - self.u + I )
            # Check which neurons are spiking.
            self.is_spike = (self.v >= self.v_peak).type(torch.float)
            # Reset all spiking neurons to the reset voltage,
            self.v += (self.v_reset - self.v) * self.is_spike
            # and increment their adaptation currents.
            # We fix b = 0 in the adaptation current equation so that we can simplify it.
            self.u = self.one_minus_dt_a * self.u + self.d * self.is_spike
            # Use the reservoir weights to integrate over incoming spikes.
            integrated_spikes = self.reservoir_weights.matmul(self.is_spike).to_dense()
            # Use a double-exponential synapse.
            self.I_synapse = self.exp_neg_dt_over_tau_r * self.I_synapse + self.dt * self.h
            self.h         = self.exp_neg_dt_over_tau_d * self.h         + self.one_over_tau_r_tau_d * integrated_spikes
            self.r         = self.exp_neg_dt_over_tau_r * self.r         + self.dt * self.hr
            self.hr        = self.exp_neg_dt_over_tau_d * self.hr        + self.one_over_tau_r_tau_d * self.is_spike
            # Take the weighted sum of r values of neurons in an area to get its prediction output.
            self.prediction = self.output_weights.matmul(self.r).to_dense()
    
    def sim_step_with_ts_input(self, ts_time_step):
        with torch.no_grad():
            self.sim_step_with_external_current( self.I_ts[:,ts_time_step].unsqueeze(1) )
    
    def sim_step(self):
        with torch.no_grad():
            self.sim_step_with_external_current(I_ext=0)
    
    # We assume correct_output is a 1D vector tensor with the same number of elements as self.prediction.
    def rls_step(self, correct_output:torch.Tensor):
        with torch.no_grad():
            Pr = self.P.matmul(self.r).to_dense().flatten()
            error_values = self.prediction.flatten() - correct_output
            self.update_output_weights(Pr, error_values)
            self.update_correlation_matrix(Pr)
    
    def update_output_weights(self, Pr:torch.Tensor, error_values:torch.Tensor):
        with torch.no_grad():
            # Replicate the error for each community to distribute a copy of it to every neuron in that community.
            # Multiply the Pr value for each neuron by the error value for its area.
            Pr_times_error = Pr * torch.repeat_interleave(error_values, self.neurons_per_community_tensor)
            # Each neuron has one non-0 output weight,
            # and its index in the COO sparse array indices and values corresponds to its index in all the other vectors.
            # self.output_weights -= torch.sparse_coo_tensor( self.output_weights._indices(), Pr_times_error, self.output_weights.size(), device=self.device )
            # This way is a little faster, since we do not need to make a new sparse matrix.
            ow_values = self.output_weights._values()
            ow_values -= Pr_times_error
    
    def update_correlation_matrix(self, Pr:torch.Tensor):
        with torch.no_grad():
            # We only want to track the correlations between r values in the same community.
            # A complete matrix of size num_neurons x num_neurons would mostly consist of correlations that could not contribute to an output.
            # To update P, we work through it one community at a time.
            # The update for each community i depends only on the relevant slice of Pr, Pr_i, and the corresponding slice of r, r_i.
            # This update is a square matrix equal to - the outer product of Pr_i with itself divided by the dot product of Pr_i with r_i.
            # Since the way we flatten the values of this matrix, delta_P_i,
            # the same way we flattened the identity matrices with which we originally filled in each block of P,
            # we can get a reference to the values in P and update the appropriate slice directly with the flattened matrix.
            block_start_index = 0
            community_start_index = 0
            r_flat = self.r.flatten()
            P_values = self.P._values()
            for i in range( self.neurons_per_community_tensor.numel() ):
                community_end_index = community_start_index + self.neurons_per_community_tensor[i]
                r_i = r_flat[community_start_index:community_end_index]
                Pr_i = Pr[community_start_index:community_end_index]
                normalization_factor = torch.dot( r_i, Pr_i ) + 1.0
                delta_P_i = torch.outer( Pr_i, Pr_i/normalization_factor )
                delta_P_i_flat = delta_P_i.flatten()
                block_size = delta_P_i_flat.numel()
                block_end_index = block_start_index+block_size
                P_values[ block_start_index:block_end_index ] -= delta_P_i_flat
                block_start_index = block_end_index
                community_start_index = community_end_index
            # delta_P = torch.sparse_coo_tensor( self.P._indices(), self.P_value_buffer, dtype=self.dtype, device=self.device )
            # self.P -= delta_P
    
    def train(self, data_ts:torch.Tensor, num_training_reps:int=1, rls_steps_per_data_step:int=1, sim_steps_per_rls_step:int=1):
        with torch.no_grad():
            num_data_time_points = data_ts.size(1)
            for _ in range(num_training_reps):
                self.reset_to_saved_state()
                self.reset_output()
                for data_step in range(1,num_data_time_points):
                    for _ in range(rls_steps_per_data_step):
                        for _ in range(sim_steps_per_rls_step):
                            sim_step_start_time = time.time()
                            self.sim_step()
                            print(f'sim step time {time.time() - sim_step_start_time:.3f}')
                        rls_step_start_time = time.time()
                        self.rls_step(data_ts[:,data_step])
                        print(f'rls step time {time.time() - rls_step_start_time:.3f}')
    
    def predict(self, num_data_time_points:int, sim_steps_per_data_step:int=1, out:torch.Tensor=None):
        with torch.no_grad():
            if out == None:
                sim_ts = torch.zeros( (self.num_communities, num_data_time_points), dtype=self.dtype, device=self.device )
            else:
                sim_ts = out
            self.reset_to_saved_state()
            self.reset_output()
            sim_ts[:,0] = self.prediction
            for data_step in range(1, num_data_time_points):
                for _ in range(sim_steps_per_data_step):
                    sim_step_start_time = time.time()
                    self.sim_step()
                    print(f'sim step time {time.time() - sim_step_start_time:.3f}')
                sim_ts[:,data_step] = self.prediction
            return sim_ts

    def make_prediction_input_weights( neurons_per_community_tensor:torch.Tensor, Q=1, dtype=torch.float, device='cpu' ):
        with torch.no_grad():
            num_communities = neurons_per_community_tensor.numel()
            num_neurons = torch.sum(neurons_per_community_tensor)
            row_indices = torch.arange( num_neurons, dtype=torch.int, device=device )
            col_indices = torch.repeat_interleave( torch.arange(num_communities, dtype=torch.int, device=device), neurons_per_community_tensor )
            values = Q * ( 2*torch.rand( num_neurons, dtype=dtype, device=device ) - 1 )
            return __class__.make_sparse_matrix( row_indices, col_indices, values, num_neurons, num_communities, device=device, compress=True )
    
    def make_const_input_weights( features_per_community: int, neurons_per_community_tensor: torch.Tensor, Q=1, dtype=torch.float, device='cpu' ):
        with torch.no_grad():
            num_communities = neurons_per_community_tensor.numel()
            num_neurons = torch.sum(neurons_per_community_tensor)
            row_indices = torch.arange(num_neurons, device=device).unsqueeze(1).repeat( (1,features_per_community) ).reshape( (-1,) )
            feature_col_offsets = torch.arange(features_per_community, device=device).unsqueeze(0).repeat( (num_neurons,1) )
            community_col_offsets = torch.repeat_interleave( features_per_community * torch.arange(num_communities, device=device), neurons_per_community_tensor ).unsqueeze(1).repeat( (1,features_per_community) )
            col_indices_mat = feature_col_offsets + community_col_offsets
            col_indices = col_indices_mat.reshape( (-1,) )
            values = Q * ( 2*torch.rand( num_neurons*features_per_community, dtype=dtype, device=device ) - 1 )
            return __class__.make_sparse_matrix(row_indices, col_indices, values, num_neurons, features_per_community * num_communities, device=device, compress=True)
    
    def make_ts_input_weights( num_ts_dimensions:int, num_neurons:int, Q:float=1.0, dtype=torch.float, device='cpu' ):
        with torch.no_grad():
            return Q * (  2*torch.rand( (num_neurons, num_ts_dimensions), dtype=dtype, device=device ) - 1  )
    
    def make_reservoir_weights( neurons_per_community_tensor:torch.Tensor, reservoir_density_tensor:torch.Tensor, G:float=1.0, dtype=torch.float, device='cpu' ):
        with torch.no_grad():
            print('in make_reservoir_weights')
            start_time = time.time()
            num_communities = neurons_per_community_tensor.numel()
            print('num_communities:',num_communities,'time: ', time.time() - start_time)
            num_neurons = torch.sum(neurons_per_community_tensor)
            print('num_neurons:',num_neurons,'time: ', time.time() - start_time)
            synapses_per_pair = torch.round( reservoir_density_tensor * neurons_per_community_tensor.unsqueeze(0) * neurons_per_community_tensor.unsqueeze(1) ).type(torch.int).flatten()
            num_synapses = torch.sum(synapses_per_pair)
            print('num_synapses:',num_synapses,'time: ', time.time() - start_time)
            community_ends = torch.cumsum(neurons_per_community_tensor, dim=0)
            community_starts = torch.roll(community_ends, 1)
            community_starts[0] = 0
            print('found community starts and community ends, time: ', time.time() - start_time)
            row_community_starts = community_starts.unsqueeze(1).repeat( (1,num_communities) ).flatten()
            col_community_starts = community_starts.unsqueeze(0).repeat( (num_communities,1) ).flatten()
            row_community_sizes = neurons_per_community_tensor.unsqueeze(1).repeat( (1,num_communities) ).flatten()
            col_community_sizes = neurons_per_community_tensor.unsqueeze(0).repeat( (num_communities,1) ).flatten()
            print('distributed community starts and ranges to community pairs, time: ', time.time() - start_time)
            row_index_start_for_synapse = torch.repeat_interleave(row_community_starts, synapses_per_pair)
            col_index_start_for_synapse = torch.repeat_interleave(col_community_starts, synapses_per_pair)
            row_index_range_for_synapse = torch.repeat_interleave(row_community_sizes, synapses_per_pair)
            col_index_range_for_synapse = torch.repeat_interleave(col_community_sizes, synapses_per_pair)
            print('distributed community starts and ranges to individual synapses, time: ', time.time() - start_time)
            row_rands = torch.rand(num_synapses, dtype=dtype, device=device)
            col_rands = torch.rand(num_synapses, dtype=dtype, device=device)
            print('generated random numbers for neuron pair selection, time: ', time.time() - start_time)
            row_indices = torch.floor( row_index_start_for_synapse + row_index_range_for_synapse * row_rands ).type(torch.int)
            col_indices = torch.floor( col_index_start_for_synapse + col_index_range_for_synapse * col_rands ).type(torch.int)
            print('rescaled random numbers to the ranges for community pairs and floored to ints, time: ', time.time() - start_time)
            values = G*torch.randn( num_synapses, dtype=dtype, device=device )
            print('found randomly selected weights, time: ', time.time() - start_time)
            weight_matrix = __class__.make_sparse_matrix( row_indices, col_indices, values, num_neurons, num_neurons, device=device, compress=True )
            print('converted indices and weights to sparse matrix, time: ', time.time() - start_time)
            return weight_matrix
            # This is the most intuitive way of picking endpoints, but it is too slow:
            # row_indices = torch.zeros(num_synapses, dtype=torch.int, device=device)
            # col_indices = torch.zeros(num_synapses, dtype=torch.int, device=device)
            # print('allocated space for indices, time: ', time.time() - start_time)
            # vec_start_index = 0
            # for i in range(num_communities):
            #     for j in range(num_communities):
            #         synapses_this_pair = synapses_per_pair[i,j]
            #         vec_end_index = vec_start_index+synapses_this_pair
            #         row_indices[ vec_start_index:vec_end_index ] = torch.randint( community_starts[i].item(), community_ends[i].item(), (synapses_this_pair,), device=device )
            #         col_indices[ vec_start_index:vec_end_index ] = torch.randint( community_starts[j].item(), community_ends[j].item(), (synapses_this_pair,), device=device )
            #         vec_start_index = vec_end_index
            # print('found randomly selected synapse endpoint indices, time: ', time.time() - start_time)
    
    def make_output_weights( neurons_per_community_tensor: torch.Tensor, dtype=torch.float, device='cpu' ):
        with torch.no_grad():
            num_communities = neurons_per_community_tensor.numel()
            num_neurons = torch.sum(neurons_per_community_tensor)
            row_indices = torch.repeat_interleave( torch.arange(num_communities, dtype=torch.int, device=device), neurons_per_community_tensor )
            col_indices = torch.arange( num_neurons, dtype=torch.int, device=device )
            values = torch.zeros( num_neurons, dtype=dtype, device=device )
            return __class__.make_sparse_matrix( row_indices, col_indices, values, num_communities, num_neurons, device=device, compress=False )
    
    def make_correlation_matrix( neurons_per_community_tensor: torch.Tensor, regularization_constant: torch.int=2, dtype=torch.float, device='cpu' ):
        with torch.no_grad():
            print('in make_correlation_matrix')
            start_time = time.time()
            num_neurons = torch.sum(neurons_per_community_tensor)
            print('num_neurons:',num_neurons,'time: ', time.time() - start_time)
            community_ends = torch.cumsum(neurons_per_community_tensor, dim=0)
            community_starts = torch.roll(community_ends, 1)
            community_starts[0] = 0
            print('found community starts and community ends, time: ', time.time() - start_time)
            row_indices = torch.cat([ torch.arange(start_index,end_index).unsqueeze(1).repeat( (1,nn.item()) ).flatten() for (start_index,end_index,nn) in zip(community_starts, community_ends, neurons_per_community_tensor) ], dim=0)
            print('found row indices of embedded identity matrices, time: ', time.time() - start_time)
            col_indices = torch.cat([ torch.arange(start_index,end_index).unsqueeze(0).repeat( (nn.item(),1) ).flatten() for (start_index,end_index,nn) in zip(community_starts, community_ends, neurons_per_community_tensor) ], dim=0)
            print('found column indices of embedded identity matrices, time: ', time.time() - start_time)
            values = torch.cat([ regularization_constant*torch.eye(cs.item(), dtype=dtype, device=device).flatten() for cs in neurons_per_community_tensor ], dim=0)
            print('found values of embedded identity matrices, time: ', time.time() - start_time)
            P = __class__.make_sparse_matrix( row_indices, col_indices, values, num_neurons, num_neurons, device=device, compress=False )
            print('converted indices and weights to sparse matrix, time: ', time.time() - start_time)
            return P
            # This way of doing it is more intuitive but slower and possibly less memory-efficient.
            # num_communities = neurons_per_community_tensor.numel()
            # block_sizes = neurons_per_community_tensor.pow(2)
            # num_elements = torch.sum(block_sizes)
            # row_indices = torch.zeros(num_elements, dtype=torch.int, device=device)
            # col_indices = torch.zeros(num_elements, dtype=torch.int, device=device)
            # values = torch.zeros(num_elements, dtype=dtype, device=device)
            # block_start_index = 0
            # for i in range(num_communities):
            #     block_size = block_sizes[i]
            #     block_end_index = block_start_index+block_size
            #     community_indices = torch.arange( community_starts[i], community_ends[i], device=device )
            #     row_indices[ block_start_index:block_end_index ] = torch.repeat_interleave( community_indices, neurons_per_community_tensor[i] )
            #     col_indices[ block_start_index:block_end_index ] = community_indices.repeat( neurons_per_community_tensor[i] )
            #     values[ block_start_index:block_end_index ] = torch.flatten( regularization_constant * torch.eye(neurons_per_community_tensor[i], device=device) )
            #     block_start_index = block_end_index

    def make_sinusoid_ts_input(num_dimensions, num_time_points, dtype=torch.float, device='cpu'):
        with torch.no_grad():
            shift_length = num_time_points//num_dimensions
            first_row = torch.zeros( num_time_points, dtype=dtype, device=device )
            first_row[0:shift_length] = torch.sin( torch.linspace(0.0, torch.pi, shift_length, dtype=dtype, device=device) )
            ts_input = torch.zeros( (num_dimensions, num_time_points), dtype=dtype, device=device )
            for d in range(num_dimensions):
                ts_input[d,:] = torch.roll( first_row, (d-1)*shift_length )
            return ts_input
    
    def make_sparse_matrix(row_indices, col_indices, values, num_rows, num_cols, device='cpu', compress=False):
        with torch.no_grad():
            indices = torch.cat(  ( row_indices.unsqueeze(0), col_indices.unsqueeze(0) ), dim=0  )
            sparse_mat = torch.sparse_coo_tensor( indices, values, (num_rows, num_cols), device=device )
            if compress:
                sparse_mat = sparse_mat.to_sparse_csr()
            return sparse_mat
    