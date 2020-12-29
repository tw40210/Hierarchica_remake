sr = 16000
timestep=0.02
batch_size=32
num_workers=8
randomsample_size=20
epoch=8
lr = 0.0001
step_to_test = 30
step_to_save = 90
whole_song_max_len= 700
gamma_mu=1/2
onoff = 12
FEAT_channel=3

label_threshold = 0.5

runs_path = "runs"
modelcode_path = "model.py"

testsample_path = "data/test_sample/wav_label"
testsample_f_path = "data/test_sample/Process_data_S1W743HP/FEAT"
