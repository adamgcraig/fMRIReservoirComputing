import os
import torch
import time
import argparse
import hcpdatautils as hcp
import reservoirutils as resrvoir

parser = argparse.ArgumentParser(description="Train a bunch of Ising models on fMRI time series data.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_batch', help="directory where we can find the time series files")
parser.add_argument("-n", "--num_supplementary_inputs", type=int, default=36, help="number of dimensions in the supplementary time series input")
parser.add_argument("-e", "--num_epochs", type=int, default=1000, help="number of times to repeat the training time series")
parser.add_argument("-p", "--print_every_seconds", type=int, default=10, help="Print the FC correlation if this many seconds have elapsed since the last printout.")
parser.add_argument("-s", "--save_every_epochs", type=int, default=100, help="save the model once every this many epochs")
parser.add_argument("-r", "--num_reps", type=int, default=100, help="number of models to train for the subject")
# parser.add_argument("-d", "--data_subset", type=str, default='training', help="which data subset to use, either training, validation or testing")
parser.add_argument("-o", "--subject_id", type=int, default=516742, help="ID of the subject on whose fMRI data we will train")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
output_directory = args.output_directory
num_supplementary_inputs = args.num_supplementary_inputs
print(f'num_supplementary_inputs={num_supplementary_inputs}')
num_epochs = args.num_epochs
print(f'num_epochs={num_epochs}')
print_every_seconds = args.print_every_seconds
print(f'print_every_seconds={print_every_seconds}')
subject_id = args.subject_id
print(f'subject_id={subject_id}')

with torch.no_grad():
    code_start_time = time.time()
    int_type = torch.int
    float_type = torch.float
    device = torch.device('cuda')
    # device = torch.device('cpu')
    # Load, normalize, binarize, and flatten the fMRI time series data.
    print('loading fMRI data...')
    data_ts = hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device)[0,:,:]
    print( 'data ts size: ', data_ts.size() )
    data_fc = hcp.get_fc(ts=data_ts)
    print( 'data fc size: ', data_fc.size() )
    num_time_points, num_regions = data_ts.size()
    model = resrvoir.IzhikevichReservoirComputer(num_inputs=num_regions+num_supplementary_inputs, num_outputs=num_regions, dtype=float_type, device=device)
    supplementary_ts = resrvoir.make_sinusoid_ts_input(num_dimensions=num_supplementary_inputs, num_time_points=num_time_points, dtype=float_type, device=device)
    for epoch in range(num_epochs):
        model.train(data_ts=data_ts, supplementary_input_ts=supplementary_ts, const_input=None, num_epochs=1, print_every_seconds=print_every_seconds)
        sim_ts, r_ts = model.predict(num_steps=num_time_points, supplementary_input_ts=supplementary_ts, const_input=None, print_every_seconds=print_every_seconds)
        sim_fc = hcp.get_fc(sim_ts)
        fc_rmse = hcp.get_triu_rmse(sim_fc, data_fc)
        fc_corr = hcp.get_triu_corr(sim_fc, data_fc)
        print(f'epoch {epoch}, RMSE {fc_rmse:.3g}, corr {fc_corr:.3g}, time {time.time()-code_start_time:.3f}')
    sim_ts, r_ts = model.predict(num_steps=num_time_points, supplementary_input_ts=supplementary_ts, const_input=None, print_every_seconds=print_every_seconds)
    sim_fc = hcp.get_fc(sim_ts)
    fc_rmse = hcp.get_triu_rmse(sim_fc, data_fc)
    fc_corr = hcp.get_triu_corr(sim_fc, data_fc)
    print(f'epoch {epoch}, RMSE {fc_rmse:.3g}, corr {fc_corr:.3g}, time {time.time()-code_start_time:.3f}')
    model_file = os.path.join(output_directory, f'{subject_id}_ts_0_epochs_{num_epochs}.pt')
    print(f'saving model to {model_file}...')
    torch.save(model, model_file)
    print(f'done, time {time.time() - code_start_time:.3g}')