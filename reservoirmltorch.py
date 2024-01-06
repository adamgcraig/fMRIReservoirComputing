# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:33:47 2023

@author: AGCRAIG
"""
# reservoir computing machine learning module

import torch
import numpy as np

#   We set the constants to their default values from 
#   https://github.com/ModelDBRepository/190565/blob/master/CODE%20FOR%20FIGURE%205%20(USER%20SUPPLIED%20SUPERVISOR)/IZFORCEMOVIE.m
#   Nicola, W., & Clopath, C. (2017).
#   Supervised learning in spiking neural networks with FORCE training.
#   Nature communications, 8(1), 1 - 15.
class IzhikevichReservoirComputerSegmented:
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
    num_segments = 360# number of distinct reservoirs
    neurons_per_segment = 1000# number of neurons in reservoir network for a single segment
    const_inputs_per_segment = 0# number of constant input values per segment
    ts_inputs_per_segment = 0# number of time series input values per segment
    reservoir_density = 0.1# connection density of reservoir network
    Q_const = 0# weighting factor for constant inputs
    Q_ts = 0# weighting factor for time series inputs
    Q_prediction = 4000# weighting factor for previous prediction
    G_intra = 5000# weighting factor of intra-reservoir connections
    G_inter = 5000# weighting factor of inter-reservoir connections
    P_regularization_constant = 2# a constant that weights the correlation matrix P in favor of self-correlation

    def __init__(self,\
         num_segments=360,\
         neurons_per_segment=1000,\
         const_inputs_per_segment=0,\
         ts_inputs_per_segment=0,\
         Q_const=0,\
         Q_ts=0,\
         Q_prediction=4000,\
         G_intra=5000,\
         G_inter=5000,\
         reservoir_density=0.1,\
         P_regularization_constant=2,\
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
         device='cpu'):
        # Store all the scalar constants.
        self.num_segments = num_segments
        self.neurons_per_segment = neurons_per_segment
        self.const_inputs_per_segment = const_inputs_per_segment
        self.ts_inputs_per_segment = ts_inputs_per_segment
        self.Q_const = Q_const
        self.Q_ts = Q_ts
        self.Q_prediction = Q_prediction
        self.G_intra = G_intra
        self.G_inter = G_inter
        self.reservoir_density = reservoir_density
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
        # Calculate some constants from the passed-in or default values of other constants.
        self.dt_over_C = self.dt / self.C
        self.dt_a = self.dt * self.a
        self.one_minus_dt_a = 1 - self.dt_a
        self.dt_a_b = self.dt_a * self.b
        self.exp_neg_dt_over_tau_d = np.exp(-self.dt / self.tau_d)
        self.exp_neg_dt_over_tau_r = np.exp(-self.dt / self.tau_r) 
        self.one_over_tau_r_tau_d = 1 / (self.tau_r * self.tau_d)
        # Declare the shapes of some of our data structures.
        state_shape = (num_segments, neurons_per_segment)
        state_zeros = torch.zeros(state_shape, dtype=torch.float, device=self.device)
        square_mat_stack_shape = (num_segments, neurons_per_segment, neurons_per_segment)
        # Randomly generate the fixed network topology.
        self.const_input_weights = self.Q_const * (  2*torch.rand( (num_segments, neurons_per_segment, const_inputs_per_segment), dtype=torch.float, device=self.device ) - 1  )# encoding weights of const input
        self.ts_input_weights = self.Q_ts * (  2*torch.rand( (num_segments, neurons_per_segment, ts_inputs_per_segment), dtype=torch.float, device=self.device ) - 1  )# encoding weights of ts input
        self.prediction_input_weights = self.Q_prediction * ( 2*torch.rand(state_shape, dtype=torch.float, device=self.device) - 1 )# encoding weights of previous prediction 
        has_intra_edge = torch.rand(square_mat_stack_shape, dtype=torch.float, device=self.device) < self.reservoir_density# Which nodes within a reservoir are connected?
        intra_edge_weight_if_any = torch.randn(square_mat_stack_shape, dtype=torch.float, device=self.device)# If they are connected, with what weight?
        self.intra_reservoir_weights = self.G_intra * has_intra_edge * intra_edge_weight_if_any# synapse weights within the reservoir
        # For now, just have a single connection with unit weight for each ordered pair of areas.
        self.inter_reservoir_weights = self.G_inter * torch.ones( (num_segments, num_segments), dtype=torch.float, device=self.device )
        # Initialize the matrices we update during FORCE training.
        self.output_weights = state_zeros.clone()# output weights used to generate prediction from r
        self.P = torch.zeros(square_mat_stack_shape, dtype=torch.float, device=self.device)# correlation matrix of r values of neurons within an area
        for i in range(0, num_segments):
            self.P[i,:,:] = self.P_regularization_constant * torch.eye(neurons_per_segment, neurons_per_segment)# "network estimate of the inverse of the correlation matrix" according to the paper
        # Set the initial neuron state vectors.
        self.v = self.v_resting + (self.v_peak - self.v_resting) * torch.rand(state_shape, dtype=torch.float, device=self.device)# mV, membrane potential 
        self.I_synapse = state_zeros.clone()# pA, post-synaptic current 
        self.u = state_zeros.clone()# pA, adaptation current 
        self.is_spike = state_zeros.clone()# 1.0 if the neuron is spiking, 0 otherwise
        self.h = state_zeros.clone()# pA/ms, synaptic current gating variable? 
        self.hr = state_zeros.clone()# pA/ms, output current gating variable? 
        self.r = state_zeros.clone()# pA, network output before transformation by output weights 
        self.prediction = torch.zeros( (num_segments), dtype=torch.float, device=self.device )# predicted value of the time series.
        # Store the initial states for v so we can reset to them later.
        self.save_reset_state()
        # If no other constant input is set, I_const is just I_bias.
        self.I_const = self.I_bias
        self.I_hdts = 0
    
    def save_reset_state(self):
        self.v_0 = self.v.clone()
        self.I_synapse_0 = self.I_synapse.clone()
        self.u_0 = self.u.clone()
        self.h_0 = self.h.clone()
        self.hr_0 = self.hr.clone()
        self.r_0 = self.r.clone()
        self.prediction_0 = self.prediction.clone()
    
    def reset(self):
        self.v = self.v_0.clone()
        self.I_synapse = self.I_synapse_0.clone()
        self.u = self.u_0.clone()
        self.h = self.h_0.clone()
        self.hr = self.hr_0.clone()
        self.r = self.r_0.clone()
        self.prediction = self.prediction_0.clone()
    
    # We assume const_input is of shape (num_segments, const_inputs_per_segment)
    def set_const_input(self, const_input):
        self.I_const = torch.sum(self.const_input_weights * const_input[:,None,:], 2) + self.I_bias
    
    def set_hdts_input(self, hdts_length):
        num_hdts_dims = self.ts_inputs_per_segment
        shift_length = hdts_length//num_hdts_dims
        first_row = torch.zeros( hdts_length, dtype=torch.float, device=self.device )
        first_row[0:shift_length] = torch.sin( torch.linspace(0.0, torch.pi, shift_length) )
        hdts_input = torch.zeros( (self.ts_inputs_per_segment, hdts_length), dtype=torch.float, device=self.device )
        for d in range(num_hdts_dims):
            hdts_input[d,:] = torch.roll( first_row, (d-1)*shift_length )
        self.I_hdts = torch.zeros( (self.num_segments, self.neurons_per_segment, hdts_length), dtype=torch.float, device=self.device)
        for t in range(hdts_length):
            hdts_input_now = hdts_input[:,t]
            self.I_hdts[:,:,t] = torch.sum( self.ts_input_weights * hdts_input_now[None,None,:], 2 )

    def sim_step_with_external_current(self, I_ext):
        I = self.I_synapse + I_ext
        # v_minus_v_resting = v - v_resting
        # v = v + dt_over_C .* ( k .* v_minus_v_resting .* (v - v_threshold) - u + I )
        # u = one_minus_dt_a .* u + dt_a_b .* v_minus_v_resting
        self.v += self.dt_over_C * ( self.k * (self.v - self.v_resting) * (self.v - self.v_threshold) - self.u + I )
        # Reset all spiking neurons to the reset voltage,
        self.is_spike = (self.v >= self.v_peak).type(torch.float)
        # print( 'spikes:', torch.sum(self.is_spike) )
        # print('neurons spiking:',np.count_nonzero(is_spike),'or',sum(is_spike))
        self.v += (self.v_reset - self.v) * self.is_spike
        # and increment their adaptation currents.
        self.u = self.one_minus_dt_a * self.u + self.d * self.is_spike
        # Within a segment, multiply each is_spike vector by the corresponding internal weight matrix.
        integrated_spikes = torch.sum( self.intra_reservoir_weights * self.is_spike[:,None,:], 2 )
        # Between segments, take is_spike for the last num_segments neurons in a segment.
        # Take the transpose so that the row for each segment now has an is_spike from every other segment.
        # Multiply each is_spike in this square matrix by a weight.
        # Add the results to the integrated_spikes totals for the first num_segments neurons in each segment.
        last_spikes = self.is_spike[:,self.neurons_per_segment-self.num_segments:self.neurons_per_segment]
        integrated_spikes[:,0:self.num_segments] += self.inter_reservoir_weights * torch.transpose(last_spikes,1,0)
        # Assume a double-exponential synapse.
        self.I_synapse = self.exp_neg_dt_over_tau_r * self.I_synapse + self.dt * self.h
        self.h         = self.exp_neg_dt_over_tau_d * self.h         + self.one_over_tau_r_tau_d * integrated_spikes
        self.r         = self.exp_neg_dt_over_tau_r * self.r         + self.dt * self.hr
        # print('mean r:', torch.mean(self.r))
        self.hr        = self.exp_neg_dt_over_tau_d * self.hr        + self.one_over_tau_r_tau_d * self.is_spike
        # Take the weighted sum of r values of neurons in an area to get its prediction output.
        self.prediction = torch.sum(self.output_weights * self.r, 1)
        # print('mean prediction:', torch.mean(self.prediction))

    def sim_step(self):
        self.sim_step_with_external_current(self.prediction_input_weights * self.prediction[:,None] + self.I_const)
    
    def sim_step_with_hdts(self, hdts_step_index):
        I_pred = self.prediction_input_weights * self.prediction[:,None]
        self.sim_step_with_external_current(I_pred + self.I_hdts[:,:,hdts_step_index] + self.I_const)

    def sim_step_with_ts_input(self, ts_input):
        I_pred = self.prediction_input_weights * self.prediction[:,None]
        I_ts = torch.sum(self.ts_input_weights * ts_input[:,None,:], 2)
        self.sim_step_with_external_current(I_pred + I_ts + self.I_const)
    
    # RLS step with passed-in values for r and error
    # r_val assumed to be the same shape as self.r, (num_segments, neurons_per_segment)
    # error_val assumed to be the same shape as self.prediction, (num_segments)
    def rls_step_with_r_and_error(self, r_val, error_val):
        Pr = torch.bmm(self.P, r_val[:,:,None])
        self.output_weights -= error_val[:,None] * Pr[:,:,0]
        self.P -= torch.bmm( Pr, torch.transpose(Pr,1,2) / ( 1 + torch.bmm(r_val[:,None,:], Pr) ) )
    
    def rls_step(self, correct_output):
        # self.rls_calc_method_test(correct_output)
        self.rls_step_with_r_and_error(self.r, self.prediction - correct_output)

    # Do multiple sim steps, then do an RLS step using the sums of errors and r values, assuming correct output is 1-dimensional.
    def sim_steps_then_rls_step_with_const_correct(self, correct_output, num_sim_steps):
        prediction_sum = torch.zeros_like(self.prediction)
        r_sum = torch.zeros_like(self.r)
        for _ in range(num_sim_steps):
            self.sim_step()
            prediction_sum += self.prediction
            r_sum += self.r
        mean_r = r_sum/num_sim_steps
        mean_error = (prediction_sum - correct_output)/num_sim_steps
        self.rls_step_with_r_and_error(mean_r, mean_error)

    # Do multiple sim steps, then do an RLS step using the sums of errors and r values.
    def sim_steps_then_rls_step(self, correct_output):
        num_sim_steps = correct_output.size(1)
        prediction_sum = torch.zeros_like(self.prediction)
        r_sum = torch.zeros_like(self.r)
        for sim_step in range(num_sim_steps):
            self.sim_step()
            prediction_sum += self.prediction
            r_sum += self.r
        mean_r = r_sum/num_sim_steps
        mean_error = ( prediction_sum - torch.sum(correct_output,1) )/num_sim_steps
        self.rls_step_with_r_and_error(mean_r, mean_error)

    # Do multiple sim steps, then return the mean prediction.
    def sim_steps_then_mean_prediction(self, num_sim_steps):
        prediction_sum = torch.zeros_like(self.prediction)
        for _ in range(num_sim_steps):
            self.sim_step()
            prediction_sum += self.prediction
        return prediction_sum/num_sim_steps
        
    # Alternate implementation of RLS step.
    # It seems to be marginally slower.
    # def rls_step(self, correct_output):
    #    Pr = torch.sum( self.P * self.r[:,None,:], 2 )
    #    error_value = self.prediction - correct_output
    #    self.output_weights -= error_value[:,None] * Pr
    #    P_outer = Pr[:,:,None] * Pr[:,None,:]
    #    P_normalizer = 1 + torch.sum(self.r * Pr, 1)
    #    self.P -= P_outer / P_normalizer[:,None,None]
        
    # Alternate implementation of RLS step.
    # It seems to be marginally slower.
    # Keeping it because there is a small difference between the two methods,
    # and I am not sure which, if either, is more accurate.
    def rls_step_with_r_and_error_multiply_and_add(self, r_val, error_val):
        Pr = torch.sum( self.P * r_val[:,None,:], 2 )
        self.output_weights -= error_val[:,None] * Pr
        P_normalizer = 1 + torch.sum(r_val * Pr, 1)
        self.P -= Pr[:,:,None] * Pr[:,None,:] / P_normalizer[:,None,None]
    
    def rls_calc_method_test(self, correct_output):
        error_value = self.prediction - correct_output
        Pr_1 = torch.sum( self.P * self.r[:,None,:], 2 )
        delta_ow_1 = error_value[:,None] * Pr_1
        P_normalizer_1 = 1 + torch.sum(self.r * Pr_1, 1)
        P_outer_1_a = Pr_1[:,:,None] * Pr_1[:,None,:] / P_normalizer_1[:,None,None]
        P_outer_1_b = (Pr_1[:,:,None] * Pr_1[:,None,:]) / P_normalizer_1[:,None,None]
        P_outer_1_c = Pr_1[:,:,None] * (Pr_1[:,None,:] / P_normalizer_1[:,None,None])
        P_outer_1_d = (Pr_1[:,:,None] / P_normalizer_1[:,None,None]) * Pr_1[:,None,:]
        Pr_2 = torch.bmm(self.P, self.r[:,:,None])
        delta_ow_2 = error_value[:,None] * Pr_2[:,:,0]
        P_normalizer_2 = 1 + torch.bmm(self.r[:,None,:], Pr_2)
        rTP_2 = torch.transpose(Pr_2,1,2)
        P_outer_2_a = Pr_2 @ rTP_2 / P_normalizer_2
        P_outer_2_b = torch.bmm( Pr_2, rTP_2 ) / P_normalizer_2
        P_outer_2_c = torch.bmm( Pr_2, rTP_2 / P_normalizer_2 )
        P_outer_2_d = torch.bmm( Pr_2 / P_normalizer_2, rTP_2 )
        print('max abs diffs')
        print( 'Pr_1 vs Pr_2', torch.max( torch.abs(Pr_2[:,:,0] - Pr_1) ) )
        print( 'delta_ow_1 vs delta_ow_2', torch.max( torch.abs(delta_ow_2 - delta_ow_1) ) )
        print( 'Pr_normalizer_1 vs Pr_normalizer_2', torch.max( torch.abs(P_normalizer_2 - P_normalizer_1) ) )
        print( 'P_outer_1_a vs P_outer_1_b', torch.max( torch.abs(P_outer_1_a - P_outer_1_b) ) )
        print( 'P_outer_1_a vs P_outer_1_c', torch.max( torch.abs(P_outer_1_a - P_outer_1_c) ) )
        print( 'P_outer_1_a vs P_outer_1_d', torch.max( torch.abs(P_outer_1_a - P_outer_1_d) ) )
        print( 'P_outer_1_a vs P_outer_2_a', torch.max( torch.abs(P_outer_1_a - P_outer_2_a) ) )
        print( 'P_outer_1_a vs P_outer_2_b', torch.max( torch.abs(P_outer_1_a - P_outer_2_b) ) )
        print( 'P_outer_1_a vs P_outer_2_c', torch.max( torch.abs(P_outer_1_a - P_outer_2_c) ) )
        print( 'P_outer_1_a vs P_outer_2_d', torch.max( torch.abs(P_outer_1_a - P_outer_2_d) ) )
        print( 'P_outer_1_b vs P_outer_1_c', torch.max( torch.abs(P_outer_1_b - P_outer_1_c) ) )
        print( 'P_outer_1_b vs P_outer_1_d', torch.max( torch.abs(P_outer_1_b - P_outer_1_d) ) )
        print( 'P_outer_1_b vs P_outer_2_a', torch.max( torch.abs(P_outer_1_b - P_outer_2_a) ) )
        print( 'P_outer_1_b vs P_outer_2_b', torch.max( torch.abs(P_outer_1_b - P_outer_2_b) ) )
        print( 'P_outer_1_b vs P_outer_2_c', torch.max( torch.abs(P_outer_1_c - P_outer_2_c) ) )
        print( 'P_outer_1_b vs P_outer_2_d', torch.max( torch.abs(P_outer_1_c - P_outer_2_d) ) )
        print( 'P_outer_1_c vs P_outer_1_d', torch.max( torch.abs(P_outer_1_c - P_outer_1_d) ) )
        print( 'P_outer_1_c vs P_outer_2_a', torch.max( torch.abs(P_outer_1_c - P_outer_2_a) ) )
        print( 'P_outer_1_c vs P_outer_2_b', torch.max( torch.abs(P_outer_1_c - P_outer_2_b) ) )
        print( 'P_outer_1_c vs P_outer_2_c', torch.max( torch.abs(P_outer_1_c - P_outer_2_c) ) )
        print( 'P_outer_1_c vs P_outer_2_d', torch.max( torch.abs(P_outer_1_c - P_outer_2_d) ) )
        print( 'P_outer_1_d vs P_outer_2_a', torch.max( torch.abs(P_outer_1_d - P_outer_2_a) ) )
        print( 'P_outer_1_d vs P_outer_2_b', torch.max( torch.abs(P_outer_1_d - P_outer_2_b) ) )
        print( 'P_outer_1_d vs P_outer_2_c', torch.max( torch.abs(P_outer_1_d - P_outer_2_c) ) )
        print( 'P_outer_1_d vs P_outer_2_d', torch.max( torch.abs(P_outer_1_d - P_outer_2_d) ) )
        print( 'P_outer_2_a vs P_outer_2_b', torch.max( torch.abs(P_outer_2_a - P_outer_2_b) ) )
        print( 'P_outer_2_a vs P_outer_2_c', torch.max( torch.abs(P_outer_2_a - P_outer_2_c) ) )
        print( 'P_outer_2_a vs P_outer_2_d', torch.max( torch.abs(P_outer_2_a - P_outer_2_d) ) )
        print( 'P_outer_2_b vs P_outer_2_c', torch.max( torch.abs(P_outer_2_b - P_outer_2_c) ) )
        print( 'P_outer_2_b vs P_outer_2_d', torch.max( torch.abs(P_outer_2_b - P_outer_2_d) ) )
        print( 'P_outer_2_c vs P_outer_2_d', torch.max( torch.abs(P_outer_2_c - P_outer_2_d) ) )
        print('end of max abs diffs')

    # Attempt at a version where we do a single RLS step
    # using the summed r*r' and error*r' matrices from multiple sim steps.
    # Seems not to work and does not save much time versus 1 RLS step per sim step.
    def sim_steps_then_rls_step_batch_cross(self, correct_output):
        num_sim_steps = correct_output.size(1)
        r_err_sum = torch.zeros_like(self.output_weights)
        r_r_sum = torch.zeros_like(self.P)
        for sim_step in range(0, num_sim_steps):
            self.sim_step()
            error_value = self.prediction - correct_output[:,sim_step]
            r_err_sum += self.r * error_value[:,None]
            r_r_sum += self.r[:,:,None] * self.r[:,None,:]
        delta_ow = torch.bmm(self.P, r_err_sum[:,:,None])
        self.output_weights -= delta_ow[:,:,0]
        rrP = torch.bmm(r_r_sum, self.P)
        rrP_inner = 1 + torch.sum( rrP, (1,2) )
        delta_P = torch.bmm(self.P, rrP)/rrP_inner[:,None,None]
        self.P -= delta_P

    # Get a string consisting of the selected name_value pairs of attributes separated by _ characters.
    # This is useful for generating strings to use in file names.
    def get_params_string(self, include_params=None):
        if include_params == None:
            # These are the ones with which we usually experiment.
            # The others already should be tuned so that they can give edge-of-chaos dynamics,
            # so long as the inputs are of the right amplitude.
            include_params = ['Q_prediction', 'Q_const', 'neurons_per_segment']
        name_value_pair_strings = [f"{name}_{getattr(self,name)}" for name in include_params]
        return '_'.join(name_value_pair_strings)
    
    def train_once(self, data_ts, sim_steps_before_training=0, rls_steps_per_data_step=1, sim_steps_per_rls_step=1):
        num_data_time_points = data_ts.size(1)
        data_ts_init = data_ts[:,0]
        self.reset()
        for _ in range(sim_steps_before_training):
            self.sim_step()
        self.prediction = data_ts_init
        for data_step in range(1,num_data_time_points):
            for _ in range(rls_steps_per_data_step):
                for _ in range(sim_steps_per_rls_step):
                    self.sim_step()
                self.rls_step(data_ts[:,data_step])
    
    def predict(self, ts_init, num_data_time_points, sim_steps_per_data_step=1, sim_steps_before_start=0):
        sim_ts = torch.zeros( (self.num_segments, num_data_time_points), device=self.device )
        sim_ts[:,0] = ts_init
        self.reset()
        for _ in range(sim_steps_before_start):
            self.sim_step()
        self.prediction = ts_init
        for data_step in range(1,num_data_time_points):
            for _ in range(sim_steps_per_data_step):
                self.sim_step()
                sim_ts[:,data_step] = self.prediction
        return sim_ts

    def train_with_validation_reps(self, \
                                   const_input, \
                                    data_ts, \
                                    training_log_file_path='training_log.csv', \
                                    sim_steps_before_training=0, \
                                    training_reps_per_time_series=1, \
                                    rls_steps_per_data_step=1, \
                                    sim_steps_per_rls_step=1, \
                                    code_start_time=0, \
                                    time_series_name='time_series', \
                                    simulated_time_series_directory=None, \
                                    output_file_name_string=None):
        import time
        import csv
        import os
        import hcpdatautils as hcp
        sim_steps_per_data_step = sim_steps_per_rls_step * rls_steps_per_data_step
        num_data_time_points = data_ts.size(1)
        self.set_const_input(const_input)
        data_fc = hcp.get_fc(data_ts)
        data_ts_init = data_ts[:,0]
        save_sim_ts = (simulated_time_series_directory != None) and (output_file_name_string != None)
        print('starting training...')
        with open(training_log_file_path, 'w', encoding='UTF8') as training_log_file:
            training_log_writer = csv.writer(training_log_file)
            training_log_writer.writerow(['time_series', 'rep', 'time', 'rmse', 'rmse_fc'])
            for rep_index in range(training_reps_per_time_series):
                self.train_once(data_ts, sim_steps_before_training, rls_steps_per_data_step, sim_steps_per_rls_step)
                sim_ts = self.predict(data_ts_init, num_data_time_points, sim_steps_per_data_step, sim_steps_before_training)
                rmse = hcp.get_ts_rmse_torch(sim_ts, data_ts)
                sim_fc = hcp.get_fc(sim_ts)
                rmse_fc = hcp.get_fc_rmse_torch(sim_fc, data_fc)
                time_since_start = time.time() - code_start_time
                training_log_writer.writerow([time_series_name, rep_index, time_since_start, rmse, rmse_fc])
                print(f'time {time_since_start:.1f} seconds, time series {time_series_name}, rep {rep_index}, RMSE {rmse:.3f}, FC RMSE {rmse_fc:.3f}')
                if save_sim_ts:
                    print('saving simulated time series...')
                    # See https://pytorch.org/docs/stable/generated/torch.save.html
                    sim_time_series_file_path = os.path.join(simulated_time_series_directory, f"{output_file_name_string}_rep_{rep_index}.pt")
                    torch.save(sim_ts, sim_time_series_file_path)
        print(f'done, time {time.time() - code_start_time:.1f}')
