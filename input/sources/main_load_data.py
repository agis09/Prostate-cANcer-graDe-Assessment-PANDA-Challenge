import os
import numpy as np
import pandas as pd
import skimage.io
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from skimage.transform import resize
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

def resize_to_square(image: np.ndarray, img_size: int = 224, color: list = [255, 255, 255]) -> np.ndarray:
    old_size = image.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    image = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image

def tile_map(image,idx1,idx2, n_tiles, image_size, tile_size, crop_size, n_split, transform=None):
    res_img=np.full((image_size,image_size,3),255)

    tile_idxs=np.arange(n_tiles)
    if transform is not None:
        np.random.shuffle(tile_idxs)
    
    cnt=0
    for i,j in zip(idx1,idx2):                    
        tmp_img=image[i*crop_size:(i+1)*crop_size,j*crop_size:(j+1)*crop_size,:]
        if tmp_img.shape[0]<crop_size or tmp_img.shape[1]<crop_size:
            tmp_img=padding_img(tmp_img,crop_size)
        
        split_idx_list=[elm for elm in range(n_split)]
        if transform is not None:
            np.random.shuffle(split_idx_list)

        for split_idx in split_idx_list:
            split_size=crop_size//int(np.sqrt(n_split))
            idx_i=split_idx//int(np.sqrt(n_split))
            idx_j=split_idx%int(np.sqrt(n_split))
            tmp=tmp_img[idx_i*split_size:(idx_i+1)*split_size,idx_j*split_size:(idx_j+1)*split_size,:]

            tmp=Image.fromarray(tmp.astype('uint8'))
            tmp=tmp.resize((tile_size,tile_size))
            tmp=np.array(tmp)
            tile_idx1=tile_idxs[cnt] // int(np.sqrt(n_tiles))
            tile_idx2=tile_idxs[cnt] % int(np.sqrt(n_tiles))
            cnt+=1

            if transform is not None:
                tmp = transform(image=tmp)['image']
            res_img[tile_idx1*tile_size:(tile_idx1+1)*tile_size,tile_idx2*tile_size:(tile_idx2+1)*tile_size]=tmp
        
        
    return res_img

class PANDADataset(Dataset):
    def __init__(self,
                 df,
                 tile_gleason,
                 image_size,
                 n_tiles,
                 image_folder,
                 tile_size,
                 crop_size,
                 n_split,
                 max_rescale=None,
                 rand=False,
                 prediction=False,
                 transform=None,
                ):

        self.df = df.reset_index(drop=True)
        self.tile_gleason=tile_gleason
        self.image_size = image_size
        self.rand = rand
        self.transform = transform
        self.n_tiles=n_tiles
        self.image_folder=image_folder
        self.crop_size=crop_size
        self.tile_size=tile_size
        self.prediction=prediction
        self.max_rescale=max_rescale
        self.n_split=n_split

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id
        
        tmp_df=self.tile_gleason[self.tile_gleason['image_id']==img_id].sort_values(by=['gleason'],ascending=False)
        idx1=tmp_df['idx1'].values[:self.n_tiles//self.n_split]
        idx2=tmp_df['idx2'].values[:self.n_tiles//self.n_split]
        
        image = skimage.io.MultiImage(os.path.join(self.image_folder,img_id)+'.tiff')[1]
        if self.prediction:
            image=crop_white(image)
        # if image.shape[0]!=self.image_size or image.shape[1]!=self.image_size:
            # image=resize(image,(self.image_size,self.image_size),anti_aliasing=False)
        
        if self.tile_size>0:
            image=tile_map(image,idx1,idx2,self.n_tiles, self.image_size, self.tile_size,self.crop_size, self.n_split,self.transform)
#         try:
#             image=tile_map(image,idx1,idx2,self.transform)
#         except:
#             print(img_id)
#         image=np.asarray(image)

        image=resize_to_square(image,self.image_size)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = image.astype(np.float32)
        image /= 255
        image = image.transpose(2, 0, 1)

        if self.prediction:
            return torch.tensor(image)
        
        else:
            label = np.zeros(5).astype(np.float32)
            label[:row.isup_grade] = 1.
            # label=np.array(row.isup_grade).astype(int)
            return torch.tensor(image), torch.tensor(label)


