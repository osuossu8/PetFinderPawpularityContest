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
    EXP_ID = '013'
    seed = 71
    epochs = 20
    folds = [0, 1, 2, 3, 4]
    N_FOLDS = 5
    LR = 1e-4
    train_bs = 8 # 32
    valid_bs = 16 # 64
    train_root = 'input/train_npy/' # 'input/train_npy/'
    test_root = 'input/test/'
    MODEL_NAME = "swin_large_patch4_window12_384" # "swin_base_patch4_window7_224"
    in_chans = 3
    ID_COL = 'Id'
    TARGET_COL = 'Pawpularity'
    TARGET_DIM = 1
    EVALUATION = 'RMSE'
    IMG_SIZE = 384 # 224 # 512 # 256 # 900
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
                A.RandomResizedCrop(CFG.IMG_SIZE, CFG.IMG_SIZE, p=0.5, scale=(0.85, 0.95)),
                A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE, p=0.5),
            ], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=15, translate_percent=(0.1, 0.1), scale=(0.9, 1.1)),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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
            features = np.load(path)
            # path = 'input/train_cropped/crop/' + self.X[item] + '.jpg'
            # features = cv2.imread(path)
            # features = cv2.cvtColor(features, cv2.COLOR_BGR2RGB)
            if CFG.get_transforms:
                features = CFG.get_transforms['train'](image=features)['image']
            features = np.transpose(features, (2, 0, 1)).astype(np.float32)
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
            features = np.transpose(features, (2, 0, 1)).astype(np.float32)

            return {
                'x': torch.tensor(features, dtype=torch.float32),
                'meta': torch.tensor(self.Meta_features[item], dtype=torch.float32),
            }


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
            
class Pet2Model(nn.Module):
    def __init__(self, model_name):
        super(Pet2Model, self).__init__()    
        
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
        self.model.head = nn.Linear(self.model.num_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.dense1 = nn.Linear(140, 64)
        self.dense2 = nn.Linear(64, CFG.TARGET_DIM)

    def forward(self, features, metas):
        x = self.model(features)
        x = self.dropout(x)
        x = torch.cat([x, metas], 1)
        x = self.dense1(x)
        x = torch.relu(x)
        output = self.dense2(x)
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


from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    return new_data, new_targets


def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    new_targets = [targets, shuffled_targets, lam]
    return data, new_targets


def mixup_criterion(preds, new_targets):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = SmoothBCEwLogits(smoothing=0.001)
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


def cutmix_criterion(preds, new_targets):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = SmoothBCEwLogits(smoothing=0.001)
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


def loss_fn(logits, targets):
    # loss_fct = nn.BCEWithLogitsLoss()
    loss_fct = SmoothBCEwLogits(smoothing=0.001)
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
        outputs = outputs.squeeze(-1)
        targets = targets / 100.0
        loss = loss_fn(outputs, targets)
        if CFG.APEX:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        outputs = torch.sigmoid(outputs) * 100.
        targets = targets * 100.
        scores.update(targets, outputs)
        tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def train_mixup_cutmix_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))

    for data in tk0:
        optimizer.zero_grad()
        inputs = data['x'].to(device)
        targets = data['y'].to(device)
        metas = data['meta'].to(device)
        if np.random.rand()<0.5:
            inputs, new_targets = mixup(inputs, targets, 0.4)
            outputs = model(inputs, metas)
            outputs = outputs.squeeze(-1)
            new_targets = [new_targets[0] / 100.0, new_targets[1] / 100.0, new_targets[2]]
            loss = mixup_criterion(outputs, new_targets) 
        else:
            inputs, new_targets = cutmix(inputs, targets, 0.4)
            outputs = model(inputs, metas)
            outputs = outputs.squeeze(-1)
            new_targets = [new_targets[0] / 100.0, new_targets[1] / 100.0, new_targets[2]]
            loss = cutmix_criterion(outputs, new_targets)

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        outputs = torch.sigmoid(outputs) * 100.
        new_targets = [new_targets[0] * 100.0, new_targets[1] * 100.0, new_targets[2]]
        scores.update(new_targets[0], outputs)
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
            outputs = outputs.squeeze(-1)
            targets = targets / 100.0
            loss = loss_fn(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            outputs = torch.sigmoid(outputs) * 100.
            targets = targets * 100.
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
    
    df = pd.read_csv("input/train_folds_5.csv")

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
train = pd.read_csv("input/train_folds_5.csv")

sims = ['13d215b4c71c3dc603cd13fc3ec80181_373c763f5218610e9b3f82b12ada8ae5',
       '5ef7ba98fc97917aec56ded5d5c2b099_67e97de8ec7ddcda59a58b027263cdcc',
       '839087a28fa67bf97cdcaf4c8db458ef_a8f044478dba8040cc410e3ec7514da1',
       '1feb99c2a4cac3f3c4f8a4510421d6f5_264845a4236bc9b95123dde3fb809a88',
       '3c50a7050df30197e47865d08762f041_def7b2f2685468751f711cc63611e65b',
       '37ae1a5164cd9ab4007427b08ea2c5a3_3f0222f5310e4184a60a7030da8dc84b',
       '5a642ecc14e9c57a05b8e010414011f2_c504568822c53675a4f425c8e5800a36',
       '2a8409a5f82061e823d06e913dee591c_86a71a412f662212fe8dcd40fdaee8e6',
       '3c602cbcb19db7a0998e1411082c487d_a8bb509cd1bd09b27ff5343e3f36bf9e',
       '0422cd506773b78a6f19416c98952407_0b04f9560a1f429b7c48e049bcaffcca',
       '68e55574e523cf1cdc17b60ce6cc2f60_9b3267c1652691240d78b7b3d072baf3',
       '1059231cf2948216fcc2ac6afb4f8db8_bca6811ee0a78bdcc41b659624608125',
       '5da97b511389a1b62ef7a55b0a19a532_8ffde3ae7ab3726cff7ca28697687a42',
       '78a02b3cb6ed38b2772215c0c0a7f78e_c25384f6d93ca6b802925da84dfa453e',
       '08440f8c2c040cf2941687de6dc5462f_bf8501acaeeedc2a421bac3d9af58bb7',
       '0c4d454d8f09c90c655bd0e2af6eb2e5_fe47539e989df047507eaa60a16bc3fd',
       '5a5c229e1340c0da7798b26edf86d180_dd042410dc7f02e648162d7764b50900',
       '871bb3cbdf48bd3bfd5a6779e752613e_988b31dd48a1bc867dbc9e14d21b05f6',
       'dbf25ce0b2a5d3cb43af95b2bd855718_e359704524fa26d6a3dcd8bfeeaedd2e',
       '43bd09ca68b3bcdc2b0c549fd309d1ba_6ae42b731c00756ddd291fa615c822a1',
       '43ab682adde9c14adb7c05435e5f2e0e_9a0238499efb15551f06ad583a6fa951',
       'a9513f7f0c93e179b87c01be847b3e4c_b86589c3e85f784a5278e377b726a4d4',
       '38426ba3cbf5484555f2b5e9504a6b03_6cb18e0936faa730077732a25c3dfb94',
       '589286d5bfdc1b26ad0bf7d4b7f74816_cd909abf8f425d7e646eebe4d3bf4769',
       '9f5a457ce7e22eecd0992f4ea17b6107_b967656eb7e648a524ca4ffbbc172c06',
       'b148cbea87c3dcc65a05b15f78910715_e09a818b7534422fb4c688f12566e38f',
       '3877f2981e502fe1812af38d4f511fd2_902786862cbae94e890a090e5700298b',
       '8f20c67f8b1230d1488138e2adbb0e64_b190f25b33bd52a8aae8fd81bd069888',
       '221b2b852e65fe407ad5fd2c8e9965ef_94c823294d542af6e660423f0348bf31',
       '2b737750362ef6b31068c4a4194909ed_41c85c2c974cc15ca77f5ababb652f84',
       '01430d6ae02e79774b651175edd40842_6dc1ae625a3bfb50571efedc0afc297c',
       '72b33c9c368d86648b756143ab19baeb_763d66b9cf01069602a968e573feb334',
       '03d82e64d1b4d99f457259f03ebe604d_dbc47155644aeb3edd1bd39dba9b6953',
       '851c7427071afd2eaf38af0def360987_b49ad3aac4296376d7520445a27726de',
       '54563ff51aa70ea8c6a9325c15f55399_b956edfd0677dd6d95de6cb29a85db9c',
       '87c6a8f85af93b84594a36f8ffd5d6b8_d050e78384bd8b20e7291b3efedf6a5b',
       '04201c5191c3b980ae307b20113c8853_16d8e12207ede187e65ab45d7def117b']
similary_images = pd.Series(sims).str.extract(r"(?P<left>\w+)_(?P<right>\w+)")
dups = similary_images['left'].values.tolist()
print(len(dups))

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
    print(trn_df.shape)
    trn_df = trn_df[~trn_df['Id'].isin(dups)].reset_index(drop=True)
    print(trn_df.shape)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-5, last_epoch=-1)

    # ====================================================
    # apex
    # ====================================================
    if CFG.APEX:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    patience = 3 # 2 # 1
    p = 0
    min_loss = 999
    best_score = np.inf

    for epoch in range(CFG.epochs):

        logger.info("Starting {} epoch...".format(epoch+1))

        start_time = time.time()

        if epoch < 1:
            train_avg, train_loss = train_mixup_cutmix_fn(model, train_dataloader, device, optimizer, scheduler)
        else:
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
    model_paths = [f'output/{CFG.EXP_ID}/fold-{i}.bin' for i in CFG.folds]

    calc_cv(model_paths)
    print('calc cv finished!!')
