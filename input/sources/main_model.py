import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from efficientnet_pytorch import model as enet
from sync_batchnorm import SynchronizedBatchNorm1d
class enetv2(nn.Module):
    def __init__(self, backbone, out_dim,pretrained_model=None):
        super(enetv2, self).__init__()
        self.enet = enet.EfficientNet.from_name(backbone)
        if pretrained_model is not None:
            self.enet.load_state_dict(torch.load(pretrained_model[backbone]))

        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        # self.batchnorm= nn.BatchNorm1d(self.enet._fc.in_features)
        self.batchnorm=SynchronizedBatchNorm1d(self.enet._fc.in_features)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)

        x=self.batchnorm(x)

        x = self.myfc(x)
        return x
    
    
def load_models(model_files,model_dir,out_dim,device, backbone):
    models = []
    for model_f in model_files:
        model_f = os.path.join(model_dir, model_f)
        # backbone = 'efficientnet-b0'
        model = enetv2(backbone, out_dim=out_dim)
        model=nn.DataParallel(model)
        model.load_state_dict(torch.load(model_f, map_location=lambda storage, loc: storage), strict=True)
        model.eval()
        model.to(device)
        models.append(model)
        print(f'{model_f} loaded!')
    return models
