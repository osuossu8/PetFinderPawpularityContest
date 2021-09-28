import albumentations as A
import albumentations.pytorch.transforms as T
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
from torch.optim import Adam, SGD
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from pathlib import Path
from typing import List

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
    EXP_ID = '003'
    seed = 71
    epochs = 20 # 3
    folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    N_FOLDS = 5
    LR = 1e-4
    train_bs = 16
    valid_bs = 32
    train_root = 'input/train_npy/' # 'input/train_npy/'
    test_root = 'input/test/'
    MODEL_NAME = "tf_efficientnet_b0_ns" # "tf_efficientnet_b1_ns"
    in_chans = 3
    ID_COL = 'Id'
    TARGET_COL = 'Pawpularity'
    TARGET_DIM = 1
    EVALUATION = 'RMSE'
    IMG_SIZE = 512 # 256 # 900
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
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE, p=1),
            A.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0,),
            T.ToTensorV2()
        ], p=1.0),
        'valid' : A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE, p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0,),
            T.ToTensorV2()
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
            features = np.load(path)
            if CFG.get_transforms:
                features = CFG.get_transforms['train'](image=features)['image']
            targets = self.y[item]
       
            return {
                'x': torch.tensor(features, dtype=torch.float32),
                'y': torch.tensor(targets, dtype=torch.float32),
                'meta': torch.tensor(self.Meta_features[item], dtype=torch.float32),
            }
          
        else:
            path = CFG.test_root + self.X[item] + '.npy'
            features = np.load(path)
            if CFG.get_transforms:
                features = CFG.get_transforms['valid'](image=features)['image']

            return {
                'x': torch.tensor(features, dtype=torch.float32),
                'meta': torch.tensor(self.Meta_features[item], dtype=torch.float32),
            }


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
            
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    
    
class Pet2Model(nn.Module):
    def __init__(self, model_name):
        super(Pet2Model, self).__init__()    
        
        # Model Encoder
        self.net = timm.create_model(model_name, pretrained=True, in_chans=CFG.in_chans)
        self.net.classifier = nn.Linear(self.net.classifier.in_features, 128)
        self.dropout = nn.Dropout(0.1)

        self.process_meta = nn.Sequential(
            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        
        self.fc = nn.Linear(128 + 8, CFG.TARGET_DIM, bias=True)
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.fc)

    def forward(self, features, metas):
        x = self.net(features)
        x = self.dropout(x)

        meta = self.process_meta(metas)
        x = torch.cat([x, meta], 1)
        output = self.fc(x)
        return output


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
    loss_fct = RMSELoss()
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
        metas = data['meta'].to(device)
        outputs = model(inputs, metas)
        loss = loss_fn(outputs, targets)
        if CFG.APEX:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(targets, outputs)
        tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


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
            metas = data['meta'].to(device)
            outputs = model(inputs, metas)
            loss = loss_fn(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def calc_cv(model_paths):
    models = []
    for model_path in model_paths:
        model = Pet2Model(CFG.MODEL_NAME)
        model.to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models.append(model)
    
    df = pd.read_csv("input/train_folds.csv")

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
                metas = data['meta'].to(device)
                output = model(inputs, metas)
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
train = pd.read_csv("input/train_folds.csv")

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
 
    model = Pet2Model(CFG.MODEL_NAME)   
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=CFG.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)

    # ====================================================
    # apex
    # ====================================================
    if CFG.APEX:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    patience = 3
    p = 0
    min_loss = 999
    best_score = np.inf

    for epoch in range(CFG.epochs):

        logger.info("Starting {} epoch...".format(epoch+1))

        start_time = time.time()

        train_avg, train_loss = train_fn(model, train_dataloader, device, optimizer, scheduler)

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
    model_paths = [
        f'output/{CFG.EXP_ID}/fold-0.bin', 
        f'output/{CFG.EXP_ID}/fold-1.bin', 
        f'output/{CFG.EXP_ID}/fold-2.bin', 
        f'output/{CFG.EXP_ID}/fold-3.bin',
        f'output/{CFG.EXP_ID}/fold-4.bin',
    ]

    calc_cv(model_paths)
    print('calc cv finished!!')
