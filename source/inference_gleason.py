import os
import skimage.io
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from glob import glob
import sys
sys.path.append('../input/sources')
from sync_batchnorm import SynchronizedBatchNorm1d
from preprocess_model import load_models
from preprocess_load_data import inference_PANDADataset
from torch.utils.data import DataLoader



crop_size=256
image_size = 256

def inference_gleason():

    data_dir = '../input/prostate-cancer-grade-assessment/'
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    image_folder = os.path.join(data_dir, 'train_images/')
    model_dir='../result/'

    batch_size = 12
    num_workers = 12
    out_dim = 6

    backborn='efficientnet-b4'

    device = torch.device('cuda')

    img_list=[]
    for img_id in tqdm(df['image_id'].values):
        img=skimage.io.MultiImage(os.path.join(image_folder,img_id)+'.tiff')[1]
        for i in range(img.shape[0]//crop_size+1):
            for j in range(img.shape[1]//crop_size+1):
                img_list.append(img_id+'_'+str(i)+'_'+str(j))

    df_cropped=pd.DataFrame({'image_id':img_list})

    model_files = [
        'efficientnet-b4_preprocess_256to256_mod_best_fold0.pth'
    ]

    models = load_models(model_files,model_dir,out_dim,device,backborn)

    dataset = inference_PANDADataset(df_cropped, image_size,crop_size,image_folder)
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


if __name__ == '__main__':
    df=inference_gleason()
    df.to_csv('../result/inference_gleason'+str(image_size)+'_mod.csv')