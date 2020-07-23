import os
import time
import skimage.io
import numpy as np
import pandas as pd
# import cv2
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from warmup_scheduler import GradualWarmupScheduler
from efficientnet_pytorch import model as enet
import albumentations
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from glob import glob
import sys
sys.path.append('../input/sources')
from sync_batchnorm import SynchronizedBatchNorm1d
from preprocess_model import enetv2
from preprocess_load_data import train_PANDADataset

DEBUG=False


crop_size=256
image_size = 256

data_dir = '../cropped_img/'
df_train = pd.read_csv(os.path.join(data_dir, 'labels_'+str(crop_size)+'_mod.csv'))
image_folder = os.path.join(data_dir, 'train_'+str(crop_size)+'_mod/')
model_dir='../efficientnet-weight/'

result_dir='../result/'

enet_type = 'efficientnet-b4'
kernel_type = enet_type+'_preprocess_'+str(crop_size)+'to'+str(image_size)+'_mod'

fold = 0
batch_size = 200
num_workers = 12
out_dim = 6
init_lr = 3e-4
warmup_factor = 10

warmup_epo = 1
n_epochs = 1 if DEBUG else 50
df_train = df_train.sample(1000).reset_index(drop=True) if DEBUG else df_train

device = torch.device('cuda')

print(image_folder)

# clearn_img=df_train['image_id'].values
# marked_img=glob('../marker-images/*')
# marked_img=[name.split('/')[2].split('.')[0] for name in marked_img]
# clearn_img=[i for i,name in enumerate(clearn_img) if name.split('_')[0] not in marked_img ]
# df_train=df_train.loc[clearn_img,:].reset_index(drop=True)

skf = StratifiedKFold(10, shuffle=True, random_state=42)
df_train['fold'] = -1
for i, (train_idx, valid_idx) in enumerate(skf.split(df_train, df_train['label'])):
    df_train.loc[valid_idx, 'fold'] = i

pretrained_model = {
    'efficientnet-b0': model_dir+'efficientnet-b0-08094119.pth',
    'efficientnet-b2': model_dir+'efficientnet-b2-27687264.pth',
    'efficientnet-b3': model_dir+'efficientnet-b3-c8376fa2.pth',
    'efficientnet-b4': model_dir+'efficientnet-b4-e116e8b3.pth',
    'efficientnet-b5': model_dir+'efficientnet-b5-586e6cc6.pth',
    'efficientnet-b7': model_dir+'efficientnet-b7-dcc49843.pth'
}


transforms_train = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
])
transforms_val = albumentations.Compose([])

# criterion = nn.BCEWithLogitsLoss()
criterion=nn.CrossEntropyLoss()

def train_epoch(loader, optimizer):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:
        
        data, target = data.to(device), target.to(device)
        loss_func = criterion
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
    return train_loss


def val_epoch(loader, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)

            # pred = logits.sigmoid().detach()
            pred=logits.argmax(1).detach()
            LOGITS.append(logits)
            PREDS.append(pred)
            TARGETS.append(target)

            val_loss.append(loss.detach().cpu().numpy())
        val_loss = np.mean(val_loss)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    # PREDS = np.argmax(torch.cat(PREDS).cpu().numpy(),axis=1)
    # TARGETS = np.argmax(torch.cat(TARGETS).cpu().numpy(),axis=1)
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    acc = (PREDS == TARGETS).mean()
    
    qwk = cohen_kappa_score(PREDS, TARGETS, weights='quadratic')

    if get_output:
        return LOGITS
    else:
        return val_loss, acc, qwk


train_idx = np.where((df_train['fold'] != fold))[0]
valid_idx = np.where((df_train['fold'] == fold))[0]

df_this  = df_train.loc[train_idx]
df_valid = df_train.loc[valid_idx]

dataset_train = train_PANDADataset(df_this , image_size, image_folder,transform=transforms_train)
dataset_valid = train_PANDADataset(df_valid, image_size, image_folder)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=RandomSampler(dataset_train), num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, sampler=SequentialSampler(dataset_valid), num_workers=num_workers)

model = enetv2(enet_type, out_dim=out_dim)
if torch.cuda.device_count() > 1:
  print("Use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model=nn.DataParallel(model)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=init_lr/warmup_factor)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs-warmup_epo)
scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

print(len(dataset_train), len(dataset_valid))

qwk_max = 0.
best_file = result_dir+f'{kernel_type}_best_fold{fold}.pth'
for epoch in range(1, n_epochs+1):
    print(time.ctime(), 'Epoch:', epoch)
    scheduler.step(epoch-1)

    train_loss = train_epoch(train_loader, optimizer)
    val_loss, acc, qwk = val_epoch(valid_loader)

    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, val loss: {np.mean(val_loss):.5f}, acc: {(acc):.5f}, qwk: {(qwk):.5f}'
    print(content)
    with open(result_dir+f'log_{kernel_type}.txt', 'a') as appender:
        appender.write(content + '\n')

    if qwk > qwk_max:
        print('score2 ({:.6f} --> {:.6f}).  Saving model ...'.format(qwk_max, qwk))
        torch.save(model.state_dict(), best_file)
        qwk_max = qwk

torch.save(model.state_dict(), os.path.join(result_dir+f'{kernel_type}_final_fold{fold}.pth'))