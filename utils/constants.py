Initialization_Server = -1
Parameter_Server = -2
Estimate_Bandwidth = 180000  # Bytes/s
SSGD_Sync_Timeout_Limit_MSec = 10000
# Init_Job_Submission_Timeout_Limit_Sec = 7
Init_Job_Submission_Timeout_Limit_Sec = 10000000
Trace_History_Folder = "./model_trace"  # model evaluation history
VERSION = "0.761"
path = "/home/FedHydro/dataset/series_data"
model_path = "/home/FedHydro/lstm_hydro_model.model"
model_drop_path = "/home/FedHydro/hydro_lstm_model_dropout.model"
model_path2 = "/home/FedHydro/hydro_lstm_model_1_input_dim.model"
local_basic_lstm = "/home/FedHydro/dataset/code_test/fed_fomo/" \
                   "model/basic_local_model.model"
