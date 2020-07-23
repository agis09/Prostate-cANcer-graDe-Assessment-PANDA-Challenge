import os
import skimage.io
import numpy as np
import pandas as pd
import torch
import albumentations

import torch.nn as nn
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from warmup_scheduler import GradualWarmupScheduler
from glob import glob
from torch.optim import lr_scheduler
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import sys
sys.path.append('../input/sources')
from main_model import enetv2
from main_learning import train_epoch, val_epoch
from main_load_data import PANDADataset
from sync_batchnorm import convert_model, DataParallelWithCallback

DEBUG=False

crop_size=256
n_split=1 # square number
n_tiles=36
tile_size=256
image_size = tile_size*int(np.sqrt(n_tiles))

data_dir = '../input/prostate-cancer-grade-assessment/'
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
df_gleason=pd.read_csv('../result/inference_gleason'+str(crop_size)+'_efnetb0.csv')


df = df_train.loc[:100] if DEBUG else df_train

model_dir = '../efficientnet-weight/'
image_folder = os.path.join(data_dir, 'train_images/')

result_dir='../result/'
batch_size = 14
num_workers = 14
out_dim = 5
init_lr = 3e-4
warmup_factor = 10
warmup_epo = 1
n_epochs = 1 if DEBUG else 40

gpu_id =0
device = torch.device(f'cuda:{gpu_id}')

enet_type = 'efficientnet-b0'
kernel_type = enet_type+'_train_'+str(crop_size)+'to'+str(image_size)

skf = StratifiedKFold(10, shuffle=True, random_state=42)

df['fold'] = -1

for i, (train_idx, valid_idx) in enumerate(skf.split(df, df['isup_grade'])):
    df.loc[valid_idx, 'fold'] = i

pretrained_model = {
    'efficientnet-b0': model_dir+'efficientnet-b0-08094119.pth',
    'efficientnet-b2': model_dir+'efficientnet-b2-27687264.pth',
    'efficientnet-b3': model_dir+'efficientnet-b3-c8376fa2.pth',
    'efficientnet-b4': model_dir+'efficientnet-b4-e116e8b3.pth',
    'efficientnet-b5': model_dir+'efficientnet-b5-586e6cc6.pth',
    'efficientnet-b7': model_dir+'efficientnet-b7-dcc49843.pth'
}

for fold in range(10):

    train_idx = np.where((df['fold'] != fold))[0]
    valid_idx = np.where((df['fold'] == fold))[0]

    df_this  = df.loc[train_idx]
    df_valid = df.loc[valid_idx]



    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
    ])




    dataset_train = PANDADataset(df_this , df_gleason, image_size, n_tiles,image_folder,
                                    tile_size,crop_size,n_split,transform=transforms_train)
    dataset_valid = PANDADataset(df_valid, df_gleason, image_size, n_tiles,image_folder,
                                    tile_size,crop_size,n_split,transform=transforms_train)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=RandomSampler(dataset_train), num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, sampler=SequentialSampler(dataset_valid), num_workers=num_workers)

    model = enetv2(enet_type, out_dim, pretrained_model)
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model=nn.DataParallel(model)
    #   model=convert_model(model)
    #   model = DataParallelWithCallback(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=init_lr/warmup_factor)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs-warmup_epo)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, 
                                        total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

    print(len(dataset_train), len(dataset_valid))

    qwk_max = 0.
    kernel_type=kernel_type


    best_file = result_dir+f'{kernel_type}_best_fold{fold}.pth'



    for epoch in range(1, n_epochs+1):
        print(time.ctime(), 'Epoch:', epoch)
        scheduler.step(epoch-1)

        train_loss = train_epoch(model,train_loader, device,optimizer)
        val_loss, acc, qwk = val_epoch(model,valid_loader,device,df_valid)

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, val loss: {np.mean(val_loss):.5f}, acc: {(acc):.5f}, qwk: {(qwk):.5f}'
        print(content)
        with open(result_dir+f'log_{kernel_type}.txt', 'a') as appender:
            appender.write(content + '\n')

        if qwk > qwk_max:
            print('score2 ({:.6f} --> {:.6f}).  Saving model ...'.format(qwk_max, qwk))
            torch.save(model.state_dict(), best_file)
            qwk_max = qwk

    torch.save(model.state_dict(), os.path.join(result_dir+f'{kernel_type}_final_fold{fold}.pth'))

