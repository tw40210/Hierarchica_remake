import preprocess
import numpy as np
import hparam
import os
from tqdm import tqdm
from keras.models import load_model
import shutil

#
raw_dir = "data/test/EvaluationFramework_ISMIR2014/DATASET"
tar_dir = "data/test/Process_data"

InFile = "data/TONAS/Deblas/01-D_AMairena.wav"

OutFile_FEAT = "FEAT.npy"
OutFile_Z = "Z.npy"
OutFile_CF = "CF.npy"
OutFile_P = "P.npy"

modelname = 'checkpoint/model3_patch25'
model = load_model(modelname)

for file in tqdm(os.listdir(raw_dir)) :
    if '.wav' in file:
        InFile = os.path.join(raw_dir, file)

        os.makedirs(os.path.join(tar_dir, "FEAT"), exist_ok=True)
        OutFile_FEAT = os.path.join(os.path.join(tar_dir, "FEAT"), f"{file}_FEAT.npy")
        os.makedirs(os.path.join(tar_dir, "Z"), exist_ok=True)
        OutFile_Z = os.path.join(os.path.join(tar_dir, "Z"), f"{file}_Z.npy")
        os.makedirs(os.path.join(tar_dir, "CF"), exist_ok=True)
        OutFile_CF = os.path.join(os.path.join(tar_dir, "CF"), f"{file}_CF.npy")
        os.makedirs(os.path.join(tar_dir, "P"), exist_ok=True)
        OutFile_P = os.path.join(os.path.join(tar_dir, "P"), f"{file}_P.npy")

        preprocess.melody_extraction(InFile, OutFile_P,model)
        preprocess.output_feature_extraction(InFile, OutFile_FEAT, OutFile_Z, OutFile_CF)


print("finish!")


#==================ISMIR 2014 audio complete
#
# src_dir="data/audio/"
# tar_dir = "data/test_audio/"
#
# shutil.copy(src_dir+'q1.wav',tar_dir+'afemale1.wav')
# shutil.copy(src_dir+'q2.wav',tar_dir+'afemale2.wav')
# shutil.copy(src_dir+'q8.wav',tar_dir+'afemale3.wav')
# shutil.copy(src_dir+'q9.wav',tar_dir+'afemale4.wav')
# shutil.copy(src_dir+'q14.wav',tar_dir+'afemale5.wav')
# shutil.copy(src_dir+'q16.wav',tar_dir+'afemale6.wav')
# shutil.copy(src_dir+'q18.wav',tar_dir+'afemale7.wav')
# shutil.copy(src_dir+'q80.wav',tar_dir+'afemale8.wav')
# shutil.copy(src_dir+'q85.wav',tar_dir+'afemale9.wav')
# shutil.copy(src_dir+'q86.wav',tar_dir+'afemale10.wav')
# shutil.copy(src_dir+'q87.wav',tar_dir+'afemale11.wav')
#
# shutil.copy(src_dir+'q21.wav',tar_dir+'amale1.wav')
# shutil.copy(src_dir+'q22.wav',tar_dir+'amale2.wav')
# shutil.copy(src_dir+'q24.wav',tar_dir+'amale3.wav')
# shutil.copy(src_dir+'q34.wav',tar_dir+'amale4.wav')
# shutil.copy(src_dir+'q56.wav',tar_dir+'amale5.wav')
# shutil.copy(src_dir+'q58.wav',tar_dir+'amale6.wav')
# shutil.copy(src_dir+'q59.wav',tar_dir+'amale7.wav')
# shutil.copy(src_dir+'q61.wav',tar_dir+'amale8.wav')
# shutil.copy(src_dir+'q62.wav',tar_dir+'amale9.wav')
# shutil.copy(src_dir+'q63.wav',tar_dir+'amale10.wav')
# shutil.copy(src_dir+'q73.wav',tar_dir+'amale11.wav')
# shutil.copy(src_dir+'q96.wav',tar_dir+'amale12.wav')
# shutil.copy(src_dir+'q102.wav',tar_dir+'amale13.wav')


#=====
