import torch
import math
import time

class IzhikevichReservoirComputer:
    # Reservoir computing model that uses Izhikevich neurons based on
    # Nicola, W., & Clopath, C. (2017). Supervised learning in spiking neural networks with FORCE training. Nature communications, 8(1), 2208.
    # Izhikevich neuron model parameters
    # dt = 0.04# ms, Euler integration step size
    # I_bias = 1000.0# pA, constant bias current
    # C = 250.0# microF, membrane capacitance
    # k = 2.5# ns/mV, scaling factor of action potential half-width
    # v_peak = 30.0# mV, peak voltage, maximum v attained during a spike
    # v_reset = -65.0# mV, reset potential, v becomes this after a spike
    # v_resting = -60.0# mV, resting v
    # v_threshold = -40.0# mV, threshold v (when b=0 and I_bias=0)
    # a = 0.002# ms^-1, reciprocal of u time constant
    # b = 0.0# nS, sensitivity of u to subthreshold oscillations of v
    # d = 100.0# pA, incremental increase in u after each spike
    # tau_d = 20.0 ms, synaptic decay time
    # tau_r = 2.0 ms, synaptic rise time
    
    # Derived Izhikevich neuron model parameters
    # dt_over_C = dt / C
    # dt_a = dt * a
    # one_minus_dt_a = 1 - dt_a
    # dt_a_b = dt_a * b
    # exp_neg_dt_over_tau_d = math.exp(-dt / tau_d)
    # exp_neg_dt_over_tau_r = math.exp(-dt / tau_r) 
    # one_over_tau_r_tau_d = 1 / (tau_r * tau_d)
    
    #  Network properties
    # num_neurons = 1000 number of neurons in the reservoir
    # num_inputs = 360 number of inputs to pass in at each time step
    # num_outputs = 360 number of outputs to produce at each time step
    # Q = 400.0 weighting factor(s) for inputs
    # Pass in a scalar float to use the same weighting factor for all inputs.
    # Pass in a Tensor of length num_inputs to give them distinct weights.
    # G = 5000.0 weighting factor of connections between neurons
    # reservoir_density=0.1 density of synaptic connections in the reservoir network
    # P_regularization_constant = 2.0 a constant that weights the correlation matrix P in favor of self-correlation

    # In general, dtype should be a floating-point datatype as reflected in the default values for the parameters,
    # but you can specify it to be 32-bit or 64-bit.

    def __init__(self,\
                 num_neurons:int = 1000,\
                 num_inputs:int = 360,\
                 num_outputs:int = 360,\
                 Q:float=400.0,\
                 G:float=5000.0,\
                 reservoir_density:float=0.1,\
                 P_regularization_constant:float=2.0,\
                 dt:float=0.04,\
                 I_bias:float=1000.0,\
                 C:float=250.0,\
                 k:float=2.5,\
                 v_peak:float=30.0,\
                 v_reset:float=-65.0,\
                 v_resting:float=-60.0,\
                 v_threshold:float=-40.0,\
                 a:float=0.002,\
                 b:float=0.0,\
                 d:float=100.0,\
                 tau_d:float=20.0,\
                 tau_r:float=2.0,\
                 device='cpu',
                 dtype=torch.float):
            # Store all the scalar constants.
            self.num_inputs = num_inputs
            self.num_neurons = num_neurons
            self.num_outputs = num_outputs
            self.Q = Q
            self.G = G
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
            self.dtype = dtype
            # Calculate some constants from the passed-in or default values of other constants.
            self.dt_over_C = self.dt / self.C
            self.dt_a = self.dt * self.a
            self.one_minus_dt_a = 1 - self.dt_a
            self.dt_a_b = self.dt_a * self.b
            self.exp_neg_dt_over_tau_d = math.exp(-self.dt / self.tau_d)
            self.exp_neg_dt_over_tau_r = math.exp(-self.dt / self.tau_r) 
            self.one_over_tau_r_tau_d = 1 / (self.tau_r * self.tau_d)
            # Randomly generate the fixed network topology.
            if ( type(self.Q) == torch.Tensor ) and (  len( self.Q.size() ) == 1  ):
                self.Q = self.Q.unsqueeze(dim=1)
            self.input_weights = self.Q * ( 2.0 * torch.rand( (self.num_inputs, self.num_neurons), dtype=self.dtype, device=self.device ) - 1.0 )
            self.reservoir_weights = self.G * torch.randn( (self.num_neurons, self.num_neurons), dtype=self.dtype, device=self.device ) * torch.bernoulli( torch.full( size=(self.num_neurons, self.num_neurons), fill_value=self.reservoir_density, dtype=self.dtype, device=self.device )  )# synapse weights within the reservoir
            self.output_weights = torch.zeros( (self.num_neurons, self.num_outputs), dtype=self.dtype, device=self.device )
            self.P = self.P_regularization_constant * torch.eye(self.num_neurons, self.num_neurons, dtype=self.dtype, device=self.device)# "network estimate of the inverse of the correlation matrix" according to the paper
            # Set the initial neuron state vectors.
            self.state_shape = (num_neurons,)
            self.v = self.v_resting + (self.v_peak - self.v_resting) * torch.rand(self.state_shape, dtype=self.dtype, device=self.device)# mV, membrane potential 
            self.state_zeros = torch.zeros(self.state_shape, dtype=self.dtype, device=self.device)
            self.I_synapse = self.state_zeros.clone()# pA, post-synaptic current 
            self.u = self.state_zeros.clone()# pA, adaptation current 
            self.is_spike = self.state_zeros.clone()# 1.0 if the neuron is spiking, 0 otherwise
            self.h = self.state_zeros.clone()# pA/ms, synaptic current gating variable? 
            self.hr = self.state_zeros.clone()# pA/ms, output current gating variable? 
            self.r = self.state_zeros.clone()# pA, network output before transformation by output weights
            # Store the initial states of the neurons so we can reset to them later.
            self.set_saved_state()
    
    # Store the current states of the neurons we can reset to them later.
    # This does not alter the learned values of P or output_weights or the current prediction value.
    def set_saved_state(self):
        self.v_0 = self.v.clone()
        self.I_synapse_0 = self.I_synapse.clone()
        self.u_0 = self.u.clone()
        self.is_spike_0 = self.is_spike.clone()
        self.h_0 = self.h.clone()
        self.hr_0 = self.hr.clone()
        self.r_0 = self.r.clone()
    
    def reset_to_saved_state(self):
        self.v = self.v_0.clone()
        self.I_synapse = self.I_synapse_0.clone()
        self.u = self.u_0.clone()
        self.is_spike = self.is_spike_0.clone()
        self.h = self.h_0.clone()
        self.hr = self.hr_0.clone()
        self.r = self.r_0.clone()
    
    # We assume input is a 1D Tensor with num_inputs elements.
    def sim_step(self, input:torch.Tensor):
        I = self.I_synapse + self.I_bias + torch.matmul(input, self.input_weights)
        v_minus_v_resting = self.v - self.v_resting
        self.v += self.dt_over_C * ( self.k * v_minus_v_resting * (self.v - self.v_threshold) - self.u + I )
        # Check which neurons are spiking.
        self.is_spike = (self.v >= self.v_peak).type(torch.float)
        # Reset all spiking neurons to the reset voltage,
        self.v += (self.v_reset - self.v) * self.is_spike
        # and increment their adaptation currents.
        self.u = self.one_minus_dt_a * self.u + self.dt_a_b * v_minus_v_resting + self.d * self.is_spike
        # Use the reservoir weights to integrate over incoming spikes.
        integrated_spikes = torch.matmul(self.is_spike, self.reservoir_weights)
        # Use a double-exponential synapse.
        self.I_synapse = self.exp_neg_dt_over_tau_r * self.I_synapse + self.dt * self.h
        self.h         = self.exp_neg_dt_over_tau_d * self.h         + self.one_over_tau_r_tau_d * integrated_spikes
        self.r         = self.exp_neg_dt_over_tau_r * self.r         + self.dt * self.hr
        self.hr        = self.exp_neg_dt_over_tau_d * self.hr        + self.one_over_tau_r_tau_d * self.is_spike
        # Take the weighted sum of r values of neurons in an area to get its prediction output.
        return torch.matmul(self.r, self.output_weights)

    # We assume correct_output is a 1D vector tensor with num_outputs elements.
    def rls_step(self, output:torch.Tensor, correct_output:torch.Tensor):
            rP = torch.matmul(self.r, self.P)
            self.output_weights -= torch.outer(rP, output-correct_output)
            self.P -= torch.outer(rP, rP)/( 1.0 + torch.dot(rP, self.r) )
    
    # For train() and predict(), we assume that num_inputs is the same as num_outputs + supplementary_input_ts.size(dim=-1) + const_input.size(dim=-1).

    def train(self, data_ts:torch.Tensor, supplementary_input_ts:torch.Tensor=None, const_input:torch.Tensor=None, num_epochs:int=1, print_every_seconds:float=60):
        start_time = time.time()
        last_print_time = start_time
        num_steps = data_ts.size(dim=-2)
        sim_ts = torch.zeros_like(data_ts)
        if type(supplementary_input_ts) != type(None):
            sim_ts = torch.cat( (sim_ts, supplementary_input_ts), dim=-1 )
        if type(const_input) != type(None):
            sim_ts = torch.cat(   (  sim_ts, const_input.unsqueeze(dim=0).repeat( (num_steps,1) )  ), dim=-1   )
        for epoch in range(num_epochs):
            for step in range(num_steps):
                output = self.sim_step(sim_ts[step,:])
                sim_ts[ (step+1) % num_steps, :self.num_outputs ] = output
                self.rls_step(output, data_ts[step,:])
                current_time = time.time()
                if current_time - last_print_time > print_every_seconds:
                    print(f'epoch {epoch}, step {step}, time {current_time-start_time:.3f}')
                    last_print_time = current_time
    
    def predict(self, num_steps:int, supplementary_input_ts:torch.Tensor=None, const_input:torch.Tensor=None, print_every_seconds:float=60):
        start_time = time.time()
        last_print_time = start_time
        sim_ts = torch.zeros( (num_steps, self.num_outputs), dtype=self.dtype, device=self.device )
        if type(supplementary_input_ts) != type(None):
            sim_ts = torch.cat( (sim_ts, supplementary_input_ts), dim=-1 )
        if type(const_input) != type(None):
            sim_ts = torch.cat(   (  sim_ts, const_input.unsqueeze(dim=0).repeat( (num_steps,1) )  ), dim=-1   )
        sim_ts = torch.cat(   (  sim_ts, torch.zeros( (1, self.num_inputs), dtype=self.dtype, device=self.device )  )   )
        r_ts = torch.zeros( (num_steps, self.num_neurons), dtype=self.dtype, device=self.device )
        is_spike_ts = torch.zeros( (num_steps, self.num_neurons), dtype=self.dtype, device=self.device )
        for step in range(num_steps):
            output = self.sim_step(sim_ts[step,:])
            r_ts[step,:] = self.r
            is_spike_ts[step,:] = self.is_spike
            sim_ts[step+1, :self.num_outputs] = output
            current_time = time.time()
            if current_time - last_print_time > print_every_seconds:
                print(f'step {step}, time {current_time-start_time:.3f}')
                last_print_time = current_time
        return sim_ts[1:num_steps+1,:self.num_outputs], r_ts, is_spike_ts

def make_sinusoid_ts_input(num_dimensions, num_time_points, dtype=torch.float, device='cpu'):
    shift_length = num_time_points//num_dimensions
    index = torch.arange(start=0, end=num_time_points, dtype=dtype, device=device)
    offset = shift_length * torch.arange(start=0, end=num_dimensions, dtype=dtype, device=device)
    offset_index = index.unsqueeze(dim=1) - offset.unsqueeze(dim=0)
    ts_input = torch.sin( torch.pi/shift_length * offset_index * (offset_index >= 0).type(torch.float) * (offset_index < shift_length).type(torch.float) )
    return ts_input