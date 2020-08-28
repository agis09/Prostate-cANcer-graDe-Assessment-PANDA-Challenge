import os
import skimage.io
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from preprocess_model import load_models
from preprocess_load_data import PANDADataset
from torch.utils.data import DataLoader
import yaml
with open('preprocess_config.yml', 'r') as yml:
    config = yaml.load(yml, Loader=yaml.SafeLoader)

data_dir = config['path']['data_dir']
df_train = pd.read_csv(os.path.join(data_dir, config['path']['df_train']))
df_test = pd.read_csv(os.path.join(data_dir, config['path']['df_test']))

model_dir = config['path']['model_dir']
image_folder = os.path.join(data_dir, config['path']['test_image_folder'])
is_test = os.path.exists(image_folder)  # IF test_images is not exists, we will use some train images.
image_folder = image_folder if is_test else os.path.join(data_dir, config['path']['train_image_folder'])

df = df_test if is_test else (df_train.loc[:10] if config['DEBUG'] else df_train)

# df = df_test if is_test else df_train[df_train['data_provider']=='karolinska'].loc[:100]
# df = df_test if is_test else df_train.loc[:100]
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
if config['DEBUG']:
    df_cropped=df_cropped.loc[:10]

model_files = [
    config['path']['preprocess_model']
]

models = load_models(model_files,model_dir,config['learning']['out_dim'],device)

dataset = PANDADataset(df_cropped, image_size,crop_size,image_folder)
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
              'idx1':idx1,
              'idx2':idx2,
              'gleason':PREDS})

if config['DEBUG']:
    tmp.to_csv('DEBUG.csv',index=False)
else:
    tmp.to_csv(config['path']['df_gleason'],index=False)