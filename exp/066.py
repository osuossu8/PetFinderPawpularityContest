import albumentations as A
# import albumentations.pytorch.transforms as T
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
    EXP_ID = '066'
    seed = 71
    epochs = 20
    folds = [0, 1, 2, 3, 4]
    N_FOLDS = 5
    LR = 2e-5 # 1e-4
    ETA_MIN = 7e-6 # 1e-5
    train_bs = 12 # 8 # 32
    valid_bs = 24 # 16 # 64
    log_interval = 32 # 100
    train_root = 'input/train_npy/' # 'input/train_npy/'
    test_root = 'input/test/'
    MODEL_NAME = "swin_large_patch4_window7_224" # "vit_large_patch32_224_in21k" # 'swin_base_patch4_window7_224'
    EMBEDDER_NAME = "tf_efficientnetv2_m_in21ft1k"
    in_chans = 3
    ID_COL = 'Id'
    TARGET_COL = 'Pawpularity'
    TARGET_DIM = 1
    EVALUATION = 'RMSE'
    IMG_SIZE = 448 # 384 # 224 # 512 # 256 # 900
    EARLY_STOPPING = True
    APEX = False # True
    DEBUG = False # True
    FEATURE_COLS = [
        'Subject Focus', 'Eyes', 'Face', 'Near', 'Action',
        'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 
        'Info', 'Blur',
    ]

CFG.get_transforms = {
        'train' : A.Compose([
            A.OneOf([
                A.RandomResizedCrop(CFG.IMG_SIZE, CFG.IMG_SIZE, p=0.3, scale=(0.85, 0.95)),
                A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE, p=0.4),
                A.Compose([
                    A.Resize(int(CFG.IMG_SIZE * 1.5), int(CFG.IMG_SIZE * 1.5), p=1.0),
                    A.CenterCrop(p=1.0, height=CFG.IMG_SIZE, width=CFG.IMG_SIZE),
                ], p=0.3),
            ], p=1.0),
            # A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            # A.Affine(rotate=15, translate_percent=(0.1, 0.1), scale=(0.9, 1.1)),
            # A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0,),
        ], p=1.0),
        'valid' : A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE, p=1),
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


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def calc_loss(y_true, y_pred):
    if CFG.EVALUATION == 'RMSE':
        return  np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    elif CFG.EVALUATION == 'AUC':
        return metrics.roc_auc_score(np.array(y_true), np.array(y_pred))
    else:
        raise NotImplementedError()


class Pet2Dataset:
    def __init__(self, X, y=None, Meta_features=None):
        self.X = X
        self.y = y
        self.Meta_features = Meta_features

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if self.y is not None:
            path = CFG.train_root + self.X[item] + '.npy'
            img = np.load(path)

            height = img.shape[0]
            width = img.shape[1]
            
            if height > width:
                center = int(height/2)
                half = int(width/2)
                img = img[center - half:center + half, :]

            elif width > height:
                center = int(width/2)
                half = int(height/2)
                img = img[:, center - half:center + half]

            if CFG.get_transforms:
                features = CFG.get_transforms['train'](image=img)['image']
            features = np.transpose(features, (2, 0, 1)).astype(np.float32)
            targets = self.y[item]
       
            return {
                'x': torch.tensor(features, dtype=torch.float32),
                'y': torch.tensor(targets, dtype=torch.float32),
                'meta': torch.tensor(self.Meta_features[item], dtype=torch.float32),
            }
          
        else:
            path = CFG.test_root + self.X[item] + '.npy'
            img = np.load(path)

            height = img.shape[0]
            width = img.shape[1]
            
            if height > width:
                center = int(height/2)
                half = int(width/2)
                img = img[center - half:center + half, :]

            elif width > height:
                center = int(width/2)
                half = int(height/2)
                img = img[:, center - half:center + half]

            if CFG.get_transforms:
                features = CFG.get_transforms['valid'](image=img)['image']
            features = np.transpose(features, (2, 0, 1)).astype(np.float32)

            return {
                'x': torch.tensor(features, dtype=torch.float32),
                'meta': torch.tensor(self.Meta_features[item], dtype=torch.float32),
            }


from functools import partial
from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, patch_size=1, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Pet2HybridModel(nn.Module):
    def __init__(self, backbone, embedder):
        super(Pet2HybridModel, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0, in_chans=CFG.in_chans)
        pretrained_model_path = '/root/.cache/torch/checkpoints/swin_large_patch4_window7_224_22kto1k.pth'
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
            self.backbone.load_state_dict(state_dict)
            print("loaded pretrained weight for backbone")
        self.embedder = timm.create_model(embedder, features_only=True, out_indices=[2], pretrained=True)
        print("loaded pretrained weight for embedder")
        self.backbone.patch_embed = HybridEmbed(self.embedder, img_size=CFG.IMG_SIZE, embed_dim=192)
        self.fc = nn.Linear(self.backbone.num_features, CFG.TARGET_DIM)

    def forward(self, images):
        features = self.backbone(images)              # features = (bs, embedding_size)
        output = self.fc(features)                    # outputs  = (bs, num_classes)
        return output.squeeze(-1)


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


class MetricMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred.cpu().detach().numpy().tolist())

    @property
    def avg(self):
        self.score = calc_loss(self.y_true, self.y_pred)
       
        return {
            CFG.EVALUATION : self.score,
        }


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


def loss_fn(logits, targets):
    loss_fct = torch.nn.BCEWithLogitsLoss()
    loss = loss_fct(logits, targets)
    return loss


def train_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for data in tk0:
        optimizer.zero_grad()
        inputs = data['x'].to(device)
        targets = data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets / 100.0)
        if CFG.APEX:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        outputs = torch.sigmoid(outputs) * 100.
        scores.update(targets, outputs)
        tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def train_fn_calc_cv_interval(epoch, model, train_data_loader, valid_data_loader, device, optimizer, scheduler, best_score):
    model.train()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(train_data_loader, total=len(train_data_loader))
    
    for batch_idx, data in enumerate(tk0):
        optimizer.zero_grad()

        inputs = data['x'].to(device)
        targets = data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets / 100.0)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        outputs = torch.sigmoid(outputs) * 100.
        scores.update(targets, outputs)
        tk0.set_postfix(loss=losses.avg)

        if (batch_idx > 0) and (batch_idx % CFG.log_interval == 0):
            valid_avg, valid_loss = valid_fn(model, valid_data_loader, device)

            logger.info(f"Epoch {epoch+1}, Step {batch_idx} - valid_score:{valid_avg[CFG.EVALUATION]:0.5f}")

            if valid_avg[CFG.EVALUATION] < best_score:
                logger.info(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg[CFG.EVALUATION]}")
                torch.save(model.state_dict(), OUTPUT_DIR+f'fold-{fold}.bin')
                best_score = valid_avg[CFG.EVALUATION]

            model.train() 

    return scores.avg, losses.avg, valid_avg, valid_loss, best_score


def valid_fn(model, data_loader, device):
    model.eval()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    valid_preds = []
    with torch.no_grad():
        for data in tk0:
            inputs = data['x'].to(device)
            targets = data['y'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets / 100.0)
            losses.update(loss.item(), inputs.size(0))
            outputs = torch.sigmoid(outputs) * 100.
            scores.update(targets, outputs)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def calc_cv(model_paths):
    models = []
    for model_path in model_paths:
        model = Pet2HybridModel(CFG.MODEL_NAME, CFG.EMBEDDER_NAME)   
        model.to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models.append(model)
    
    df = pd.read_csv("input/train_folds_no_dup_5.csv")

    idx = []
    y_true = []
    y_pred = []
    for fold, model in enumerate(models):
        val_df = df[df.kfold == fold].reset_index(drop=True)
    
        dataset = Pet2Dataset(X=val_df[CFG.ID_COL].values, y=val_df[CFG.TARGET_COL].values, Meta_features=val_df[CFG.FEATURE_COLS].values)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=CFG.valid_bs, num_workers=0, pin_memory=True, shuffle=False
        )

        final_output = []
        for b_idx, data in tqdm(enumerate(data_loader)):
            with torch.no_grad():
                inputs = data['x'].to(device)
                targets = data['y'].to(device)
                output = model(inputs)
                output = torch.sigmoid(output) * 100.
                output = output.detach().cpu().numpy().tolist()
                final_output.extend(output)
        y_pred.append(np.array(final_output))
        y_true.append(val_df[CFG.TARGET_COL].values)
        idx.append(val_df[CFG.ID_COL].values)
        torch.cuda.empty_cache()
        
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    idx = np.concatenate(idx)
    overall_cv_score = calc_loss(y_true, y_pred)
    logger.info(f'cv score {overall_cv_score}')
    oof_df = pd.DataFrame()
    oof_df[CFG.ID_COL] = idx
    oof_df['oof'] = y_pred
    oof_df[CFG.TARGET_COL] = y_true
    oof_df.to_csv(OUTPUT_DIR+"oof.csv", index=False)
    print(oof_df.shape)


OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

warnings.filterwarnings("ignore")
logger = init_logger(log_file=Path("log") / f"{CFG.EXP_ID}.log")

# environment
set_seed(CFG.seed)
device = get_device()

# data
train = pd.read_csv("input/train_folds_no_dup_5.csv")

print(train.shape)
train.head()

# main loop
for fold in range(5):
    if fold not in CFG.folds:
        continue
    logger.info("=" * 120)
    logger.info(f"Fold {fold} Training")
    logger.info("=" * 120)

    trn_df = train[train.kfold != fold].reset_index(drop=True)
    val_df = train[train.kfold == fold].reset_index(drop=True)

    if CFG.DEBUG:
        trn_df = trn_df.head(64)
        val_df = val_df.head(16)

    train_dataset = Pet2Dataset(X=trn_df[CFG.ID_COL].values, y=trn_df[CFG.TARGET_COL].values, Meta_features=trn_df[CFG.FEATURE_COLS].values)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.train_bs, num_workers=0, pin_memory=True, shuffle=True
    )
    
    valid_dataset = Pet2Dataset(X=val_df[CFG.ID_COL].values, y=val_df[CFG.TARGET_COL].values, Meta_features=val_df[CFG.FEATURE_COLS].values)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=CFG.valid_bs, num_workers=0, pin_memory=True, shuffle=False
    )
 
    model = Pet2HybridModel(CFG.MODEL_NAME, CFG.EMBEDDER_NAME)   
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=CFG.ETA_MIN, last_epoch=-1)

    # ====================================================
    # apex
    # ====================================================
    if CFG.APEX:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    patience = 3 # 4
    p = 0
    min_loss = 999
    best_score = np.inf

    for epoch in range(CFG.epochs):

        logger.info("Starting {} epoch...".format(epoch+1))

        start_time = time.time()

        train_avg, train_loss, valid_avg, valid_loss, best_score = train_fn_calc_cv_interval(epoch, model, train_dataloader, valid_dataloader, device, optimizer, scheduler, best_score)

        valid_avg, valid_loss = valid_fn(model, valid_dataloader, device)
        scheduler.step()
        
        elapsed = time.time() - start_time

        logger.info(f'Epoch {epoch+1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s')
        logger.info(f"Epoch {epoch+1} - train_score:{train_avg[CFG.EVALUATION]:0.5f}  valid_score:{valid_avg[CFG.EVALUATION]:0.5f}")        

        if valid_avg[CFG.EVALUATION] < best_score:
            logger.info(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg[CFG.EVALUATION]}")
            torch.save(model.state_dict(), OUTPUT_DIR+f'fold-{fold}.bin')
            best_score = valid_avg[CFG.EVALUATION]
            p = 0 

        if CFG.EARLY_STOPPING:
            p += 1
            if p > patience:
                logger.info(f'Early Stopping')
                break

if len(CFG.folds) == 1:
    pass
else:
    model_paths = [f'output/{CFG.EXP_ID}/fold-{i}.bin' for i in CFG.folds]

    calc_cv(model_paths)
    print('calc cv finished!!')
