sr = 16000
timestep=0.02
batch_size=32
num_workers=8
randomsample_size=20
epoch=30
lr = 0.0001
step_to_test = 20
step_to_save = 120
whole_song_max_len= 700


label_threshold = 0.5

runs_path = "runs"
modelcode_path = "model.py"

testsample_path = "data/test_sample/wav_label"
testsample_f_path = "data/test_sample/FEAT"
test_path = "data/test/EvaluationFramework_ISMIR2014/DATASET"
test_f_path = "data/test/Process_data/FEAT"