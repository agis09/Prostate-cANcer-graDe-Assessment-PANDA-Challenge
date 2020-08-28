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

from torch.optim import lr_scheduler
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.append('../input/sources')
from preprocess_model import load_models as prep_load_models
from preprocess_load_data import PANDADataset as prep_PANDAdataset

from sync_batchnorm import convert_model, DataParallelWithCallback
from main_model import enetv2
from main_model import load_models
from main_load_data import PANDADataset
import yaml

config_path='../input/configs/'

def inference_gleason(config_path):

    with open(config_path, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.SafeLoader)
    data_dir = config['path']['data_dir']
    df_train = pd.read_csv(os.path.join(data_dir, config['path']['df_train']))
    df_test = pd.read_csv(os.path.join(data_dir, config['path']['df_test']))

    model_dir = config['path']['model_dir']
    image_folder = os.path.join(data_dir, config['path']['test_image_folder'])
    is_test = os.path.exists(image_folder)  # IF test_images is not exists, we will use some train images.
    image_folder = image_folder if is_test else os.path.join(data_dir, config['path']['train_image_folder'])

    df = df_test if is_test else df_train.loc[:10]

    image_size = config['values']['image_size']
    batch_size = config['learning']['batch_size']
    num_workers = config['learning']['num_workers']
    out_dim = config['learning']['out_dim']

    gpu_id = config['device']['gpu_id']
    device = torch.device(f'cuda:{gpu_id}')

    crop_size=config['values']['crop_size']
    img_list=[]
    for img_id in tqdm(df['image_id'].values):
        img=skimage.io.MultiImage(os.path.join(image_folder,img_id)+'.tiff')
        for i in range(img[config['values']['tiff_layer']].shape[0]//crop_size+1):
            for j in range(img[config['values']['tiff_layer']].shape[1]//crop_size+1):
                img_list.append(img_id+'_'+str(i)+'_'+str(j))

    df_cropped=pd.DataFrame({'image_id':img_list})
    if not is_test:
        df_cropped=df_cropped.loc[:10]

    model_files = [
        config['path']['preprocess_model']
    ]

    models = prep_load_models(model_files,model_dir,config['learning']['out_dim'],device)

    dataset = prep_PANDAdataset(df_cropped, image_size,crop_size,image_folder)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    LOGITS = []
    PREDS=[]
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            logits = models[0](data)
            LOGITS.append(logits)
            pred=logits.argmax(1).detach()
            PREDS.append(pred)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    PREDS = torch.cat(PREDS).cpu().numpy()

    imgid=[imgid.split('_')[0] for imgid in df_cropped['image_id'].values]
    idx1=[imgid.split('_')[1] for imgid in df_cropped['image_id'].values]
    idx2=[imgid.split('_')[2] for imgid in df_cropped['image_id'].values]

    tmp=pd.DataFrame({'image_id':imgid,
                'idx1':np.array(idx1).astype(int),
                'idx2':np.array(idx2).astype(int),
                'gleason':PREDS})

    return tmp


with open(config_path+'inference_config.yml', 'r') as yml:
    config = yaml.load(yml, Loader=yaml.SafeLoader)

data_dir = config['path']['data_dir']
df_train = pd.read_csv(os.path.join(data_dir, config['path']['df_train']))
df_test = pd.read_csv(os.path.join(data_dir, config['path']['df_test']))
df_sub = pd.read_csv(os.path.join(data_dir, config['path']['df_submit']))

image_folder = os.path.join(data_dir, config['path']['test_image_folder'])
is_test = os.path.exists(image_folder)  # IF test_images is not exists, we will use some train images.
image_folder = image_folder if is_test else os.path.join(data_dir, config['path']['train_image_folder'])

df = df_test if is_test else df_train.loc[:10]

image_size = config['values']['image_size']
batch_size = config['learning']['batch_size']
num_workers = config['learning']['num_workers']
out_dim = config['learning']['out_dim']

gpu_id = config['device']['gpu_id']
device = torch.device(f'cuda:{gpu_id}')

model_files = [
    config['path']['pretrained_model']
]
model_dir = config['path']['model_dir']


df_gleason=inference_gleason(config_path+'inference_preprocess_config.yml')

print(df_gleason.head())

models = load_models(model_files,model_dir,config['learning']['out_dim'],device,config['name']['enet_type'])


dataset = PANDADataset(df, df_gleason, image_size, config['values']['n_tiles'],image_folder,
                        config['values']['tile_size'], config['values']['crop_size'],prediction=True,transform=None)


loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

LOGITS = []
PREDS=[]
with torch.no_grad():
    for data in tqdm(loader):
        data = data.to(device)
        logits = models[0](data)
        LOGITS.append(logits)

LOGITS = torch.cat(LOGITS).sigmoid().cpu() 
PREDS = LOGITS.sum(1).round().numpy()


df['isup_grade'] = PREDS.astype(int)
df[['image_id', 'isup_grade']].to_csv('submission.csv', index=False)

# print(df.isup_grade.value_counts())