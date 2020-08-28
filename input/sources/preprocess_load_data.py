import os
import numpy as np
import pandas as pd
import skimage.io

import torch
from torch.utils.data import DataLoader, Dataset
from skimage.transform import rescale,resize
from PIL import Image


def padding_img(img_tmp,crop_size):
        pad=crop_size-img_tmp.shape[0]
        pad=np.full((pad,img_tmp.shape[1],img_tmp.shape[2]),255)
        img_tmp=np.concatenate([img_tmp,pad],axis=0)
        pad=crop_size-img_tmp.shape[1]
        pad=np.full((img_tmp.shape[0],pad,img_tmp.shape[2]),255)
        img_tmp=np.concatenate([img_tmp,pad],axis=1)
        return img_tmp


def crop_white(image: np.ndarray, value: int = 255) -> np.ndarray:
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) < value).nonzero()
    xs, = (image.min((0, 2)) < value).nonzero()
    
    # if there's no pixel with such a value
    if len(xs) == 0 or len(ys) == 0:
        return image
    
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


class inference_PANDADataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 crop_size,
                 image_folder,
                 max_rescale=None,
                 transform=None,
                ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform
        self.crop_size=crop_size
        self.image=None
        self.pre_img_id=None
        self.image_folder=image_folder
        self.max_rescale=max_rescale

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id
        i=int(img_id.split('_')[1])
        j=int(img_id.split('_')[2])
        img_id=img_id.split('_')[0]
        
        if img_id != self.pre_img_id:
            self.pre_img_id=img_id
            self.image=os.path.join(self.image_folder, img_id)
            self.image = skimage.io.MultiImage(os.path.join(self.image_folder, f'{img_id}.tiff'))[1]
            self.image=crop_white(self.image)
            if self.max_rescale is not None:
                self.image=rescale(self.image,self.max_rescale/self.image.shape[0]*self.image.shape[1],anti_aliasing=False)
        
        image=self.image[i*self.crop_size:(i+1)*self.crop_size,j*self.crop_size:(j+1)*self.crop_size,:]
        if image.shape[0]<self.crop_size or image.shape[1]<self.crop_size:
                image=padding_img(image,self.crop_size)
        
        image=Image.fromarray(image.astype('uint8'))
        # image=rescale(image,image.shape[0]/self.image_size,anti_aliasing=False)
        image=np.asarray(image)
        
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = image.astype(np.float32)
        image /= 255
        image = image.transpose(2, 0, 1)

        return torch.tensor(image)

class train_PANDADataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 image_folder,
                 transform=None,
                ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.image_folder=image_folder
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id
        
        cropped_img=os.path.join(self.image_folder, img_id)
        image = skimage.io.imread(cropped_img) #.png
        if image.shape[0]!=self.image_size:
            image=resize(image,(self.image_size,self.image_size),anti_aliasing=False)
        image=np.asarray(image)
        
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = image.astype(np.float32)
        image /= 255
        image = image.transpose(2, 0, 1)

        # label = np.zeros(5).astype(np.float32)
        # label[row.label] = 1.
        label=np.array(row.label).astype(int)
        return torch.tensor(image), torch.tensor(label)