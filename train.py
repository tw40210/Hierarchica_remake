import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from utils import read_notefile, note2timestep, get_accuracy, whole_song_sampletest, Logger, get_Resnet, testset_evaluation
import hparam
import random
from tqdm import tqdm
from dataset import mydataset
from model import ResNet, BasicBlock, get_BCE_loss
from torch import optim
from tensorboardX import SummaryWriter
import mir_eval

SEED=0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_path = 'data/train/TONAS/Deblas'
train_f_path = 'data/train/Process_data/FEAT'

test_path = 'data/test/EvaluationFramework_ISMIR2014/DATASET'
test_f_path = 'data/test/Process_data/FEAT'




def train():

    train_dataloader = DataLoader(mydataset(train_path, train_f_path, amount=10000), batch_size=hparam.batch_size, shuffle=True,
                            num_workers=hparam.num_workers)
    test_dataloader = DataLoader(mydataset(hparam.test_path, hparam.test_f_path), batch_size=hparam.batch_size, shuffle=True,
                            num_workers=hparam.num_workers)


    # model = ResNet(BasicBlock, [2, 2, 2, 2])
    # num_fout = model.conv1.out_channels
    # model.conv1 = nn.Conv2d(3, num_fout, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
    #                            bias=False)
    # model.fc = nn.Linear(model.fc.in_features, 6)
    # model.avgpool = nn.AvgPool2d(kernel_size=(17, 1), stride=1, padding=0)
    model= get_Resnet().to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=hparam.lr, weight_decay=0, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    model.train()

    logger = Logger(runs_dir=hparam.runs_path)

    step_count =0
    ori_runlist = logger.get_runsdir()
    with SummaryWriter() as writer:
        new_rundir = logger.get_new_runsdir(ori_runlist)
        logger.save_codebackup(hparam.modelcode_path, new_rundir)
        for epoch in range(hparam.epoch):
            bar = tqdm(train_dataloader)
            for features_full, label_note in bar :
                features_full=features_full.to(device)
                label_note = label_note.to(device)
                for clip_id in tqdm(range(features_full.shape[-1]-18)):
                    features_full_clip = features_full[:,:,:,clip_id:clip_id+19]
                    label_note_clip = label_note[:,clip_id:clip_id+1,:]

                    optimizer.zero_grad()

                    out_label = model(features_full_clip)
                    loss = get_BCE_loss(out_label, label_note_clip)

                    loss.backward()
                    optimizer.step()
                step_count+=1

                if step_count%hparam.step_to_test ==0:
                    test_sumloss = 0
                    acc_sumloss = 0
                    batch_count = 0


                    for features_full, label_note in test_dataloader:
                        features_full = features_full.to(device)
                        label_note = label_note.to(device)
                        for clip_id in range(features_full.shape[-1] - 18):
                            features_full_clip = features_full[:, :, :, clip_id:clip_id + 19]
                            label_note_clip = label_note[:, clip_id:clip_id + 1, :]

                            out_label = model(features_full_clip)


                            test_loss = get_BCE_loss(out_label, label_note_clip)
                            test_sumloss+=test_loss
                            test_acc = get_accuracy(out_label, label_note_clip)

                            acc_sumloss+=test_acc
                            batch_count+=1


                    avg_loss = test_sumloss/batch_count
                    avg_acc = acc_sumloss / batch_count
                    writer.add_scalar(f'scalar/acc', avg_acc, step_count)
                    writer.add_scalars(f'scalar/loss', {'train_loss':loss, 'test_loss':avg_loss}, step_count)

                    bar.set_postfix({'loss':f' {loss} ' , 'test_loss': f' {avg_loss} ', 'test_acc': f' {avg_acc} '})
                    bar.update(1)

                if step_count % hparam.step_to_save == 0:
                    testsample_path = hparam.testsample_path
                    testsample_f_path = hparam.testsample_f_path
                    whole_song_sampletest(testsample_path, testsample_f_path, model=model, writer_in=writer, timestep=step_count)
                    torch.save(model.state_dict(), f"checkpoint/{step_count}.pth")
                    logger.save_modelbackup(model, new_rundir)
                    print(f'saved in {step_count}\n')

                    test_path = hparam.test_path
                    test_f_path = hparam.test_f_path

                    testset_evaluation(test_path, test_f_path, model=model, writer_in=writer, timestep=step_count)



if __name__ == '__main__':
    train()