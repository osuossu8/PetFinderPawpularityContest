import albumentations as A
import cv2
import gc
import os
import math
import random
import time
import warnings
import sys
sys.path.append("/root/workspace/PetFinderPawpularityContest")

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import transformers
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam, SGD
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from pathlib import Path
from typing import List
from PIL import Image

from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold,StratifiedKFold

import timm
from tqdm import tqdm

from apex import amp


class CFG:
    ######################
    # Globals #
    ######################
    EXP_ID = 'ssl_001'
    seed = 71
    epochs = 20
    folds = [0, 1, 2, 3, 4]
    N_FOLDS = 5
    LR = 2e-5
    ETA_MIN = 7e-6
    train_bs = 4 # 8
    valid_bs = 8 # 16
    train_root = 'input/train_npy/'
    test_root = 'input/test/'
    MODEL_NAME = "swin_large_patch4_window12_384" # "swin_base_patch4_window7_224"
    in_chans = 3
    ID_COL = 'ImagePath' # 'Id'
    TARGET_COL = 'Pawpularity'
    TARGET_DIM = 1
    EVALUATION = 'RMSE'
    IMG_SIZE = 384
    EARLY_STOPPING = True
    APEX = False # True
    DEBUG = False # True

CFG.get_transforms = {
        'train' : A.Compose([
            A.RandomResizedCrop(CFG.IMG_SIZE, CFG.IMG_SIZE, p=1, scale=(0.2, 1.)),           
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=15, translate_percent=(0.1, 0.1), scale=(0.9, 1.1)),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0,),
        ], p=1.0),
    }

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# environment
set_seed(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


class SimSiamDataset:
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        path = self.X[item]
        features = cv2.imread(path)
        features = cv2.cvtColor(features, cv2.COLOR_BGR2RGB)
        if CFG.get_transforms:
            features1 = CFG.get_transforms['train'](image=features)['image']
            features2 = CFG.get_transforms['train'](image=features)['image']

        features1 = np.transpose(features1, (2, 0, 1)).astype(np.float32)
        features2 = np.transpose(features2, (2, 0, 1)).astype(np.float32)

        return {
            'x1': torch.tensor(features1, dtype=torch.float32),
            'x2': torch.tensor(features2, dtype=torch.float32),
        }


# https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
class SimSiam(nn.Module):
    def __init__(self, model_name, dim=2048, pred_dim=512):
        super(SimSiam, self).__init__()    
        
        # Model Encoder
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0, in_chans=CFG.in_chans)
        pretrained_model_path = '/root/.cache/torch/checkpoints/swin_large_patch4_window12_384_22kto1k.pth'
        if pretrained_model_path:
            state_dict = dict()
            for k, v in torch.load(pretrained_model_path, map_location='cpu')["model"].items():
                if k[:6] == "model.":
                    k = k.replace("model.", "")
                if k == 'head.weight':
                    continue
                if k == 'head.bias':
                    continue
                state_dict[k] = v
            self.model.load_state_dict(state_dict)
            print("loaded pretrained weight")

        prev_dim = self.model.num_features
        self.model.head = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=True),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        self.model.head[6].bias.requires_grad = False 

        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer


    def forward(self, x1, x2):
        z1 = self.model(x1) # NxC
        z2 = self.model(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()


# ====================================================
# Training helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def loss_fn(p1, p2, z1, z2):
    criterion = nn.CosineSimilarity(dim=1)
    loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
    return loss


def train_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for data in tk0:
        optimizer.zero_grad()
        inputs1 = data['x1'].to(device)
        inputs2 = data['x2'].to(device)
        p1, p2, z1, z2 = model(inputs1, inputs2)
        loss = loss_fn(p1, p2, z1, z2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), inputs1.size(0))
        tk0.set_postfix(loss=losses.avg)
    return losses.avg


OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

warnings.filterwarnings("ignore")
logger = init_logger(log_file=Path("log") / f"{CFG.EXP_ID}.log")

# data
train = pd.read_csv("input/petfinder1_dataset_exist_image_path_with_pred_meta_data.csv").iloc[:-3858]
train['ImagePath'] = train['PetID'].map(lambda x: f'input/train_images/{x}-1.jpg')

print(train.shape)
train.head()

train_dataset = SimSiamDataset(X=train[CFG.ID_COL].values)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=0, pin_memory=True, shuffle=True)
    
model = SimSiam(CFG.MODEL_NAME)   
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.epochs, T_mult=1, eta_min=CFG.ETA_MIN, last_epoch=-1)

patience = 6
p = 0
min_loss = 999

for epoch in range(CFG.epochs):

    logger.info("Starting {} epoch...".format(epoch+1))

    start_time = time.time()

    train_loss = train_fn(model, train_dataloader, device, optimizer, scheduler)

    scheduler.step()
        
    elapsed = time.time() - start_time

    logger.info(f'Epoch {epoch+1} - avg_train_loss: {train_loss:.5f} time: {elapsed:.0f}s')

    if train_loss < min_loss:
        logger.info(f">>>>>>>> Model Improved From {min_loss} ----> {train_loss}")
        torch.save(model.state_dict(), OUTPUT_DIR+f'fold-{fold}.bin')
        min_loss = train_loss
        p = 0

    if CFG.EARLY_STOPPING:
        p += 1
        if p > patience:
            logger.info(f'Early Stopping')
            break

