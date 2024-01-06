rmse_table_dir = 'E:\rmse_tables\';

rmse_table_file = [rmse_table_dir 'rmse_table_time_series_516742_1_LR_reps_10000_pre_training_0_rls_per_data_1_sim_per_rls_1_nps_720_Q_const_0.0_Q_pred_5000.0_G_inter_5000.0_norm_std-mean.csv.csv'];
rmse_table = readtable(rmse_table_file);

figure
plot(rmse_table.rep, rmse_table.rmse, '-r', rmse_table.rep, rmse_table.rmse_fc, '--g')
xlabel('training pass')
ylabel('RMSE')
legend({'TS RMSE', 'FC RMSE'})
xlim([0 200])
ylim([0 1])

rmse_table_file = [rmse_table_dir 'rmse_table_train_516742_1_LR_then_100206_1_LR_reps_4000_pre_training_0_rls_per_data_1_sim_per_rls_1_nps_720_Q_const_5000.0_Q_pred_5000.0_G_inter_5000.0_norm_std-mean_dt_0.04.csv'];
rmse_table = readtable(rmse_table_file);

figure
start_index = 1;
cutoff_index = 4000;
plot(rmse_table.rep(start_index:cutoff_index), rmse_table.rmse_1(start_index:cutoff_index), '-r', rmse_table.rep(start_index:cutoff_index), rmse_table.rmse_fc_1(start_index:cutoff_index), '--g')
xlabel('training pass')
ylabel('RMSE')
legend({'TS RMSE', 'FC RMSE'})
xlim([0 200])
ylim([0 1])

figure
start_index = 4001;
cutoff_index = 7625;
plot(rmse_table.rep(start_index:cutoff_index), rmse_table.rmse_2(start_index:cutoff_index), '-r', rmse_table.rep(start_index:cutoff_index), rmse_table.rmse_fc_2(start_index:cutoff_index), '--g')
xlabel('training pass')
ylabel('RMSE')
legend({'TS RMSE', 'FC RMSE'})
xlim([0 200])
ylim([0 1])

rmse_table_file = [rmse_table_dir 'rmse_table_train_516742_1_LR_then_100206_1_LR_reps_2000_pre_training_0_rls_per_data_1_sim_per_rls_1_nps_2000_Q_const_5000.0_Q_pred_5000.0_G_inter_5000.0_norm_std-mean_dt_0.04.csv'];
rmse_table = readtable(rmse_table_file);

figure
start_index = 1;
cutoff_index = 2000;
plot(rmse_table.rep(start_index:cutoff_index), rmse_table.rmse_1(start_index:cutoff_index), '-r', rmse_table.rep(start_index:cutoff_index), rmse_table.rmse_fc_1(start_index:cutoff_index), '--g')
xlabel('training pass')
ylabel('RMSE')
legend({'TS RMSE', 'FC RMSE'})
xlim([0 200])
ylim([0 1])

figure
start_index = 2001;
cutoff_index = 4000;
plot(rmse_table.rep(start_index:cutoff_index), rmse_table.rmse_2(start_index:cutoff_index), '-r', rmse_table.rep(start_index:cutoff_index), rmse_table.rmse_fc_2(start_index:cutoff_index), '--g')
xlabel('training pass')
ylabel('RMSE')
legend({'TS RMSE', 'FC RMSE'})
xlim([0 200])
