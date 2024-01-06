#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --error=results/errors/segmented_reservoir_computing_interleave_test_%j.err
#SBATCH --output=results/outs/segmented_reservoir_computing_interleave_test_%j.out
srun python segmented_reservoir_computing_interleave_test.py --data_directory data --simulated_time_series_directory results/sim_ts_binaries --trained_model_directory trained_models --training_log_directory results/rmse_tables  --neurons_per_segment 1000 --subject_id_1 516742 --subject_id_2 100206 --time_series_suffix_1 1_LR --time_series_suffix_2 1_LR --rls_steps_per_data_step 1 --sim_steps_per_rls_step 1 --training_reps_per_time_series 10000 --inter_reservoir_weight 5000.0 --prediction_input_weight 5000.0 --const_input_weight 5000.0 --dt 0.04