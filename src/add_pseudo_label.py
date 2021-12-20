import sys
sys.path.append("/root/workspace/PetFinderPawpularityContest")

import albumentations as A
import cv2
import gc
import os
import math
import random
import time
import warnings
import sys

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

from tqdm import tqdm


class CFG:
    ######################
    # Globals #
    ######################
    seed = 71
    epochs = 20
    folds = [0, 1, 2, 3, 4]
    N_FOLDS = 5
    LR = 1e-4
    train_bs = 16
    valid_bs = 32
    train_root = '../input/petfinder-pawpularity-score/train/'
    test_root = '../input/petfinder-pawpularity-score/test/'
    in_chans = 3
    ID_COL = 'Id'
    TARGET_COL = 'Pawpularity'
    TARGET_DIM = 1
    FEATURE_COLS = [
        'Subject Focus', 'Eyes', 'Face', 'Near', 'Action',
        'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 
        'Info', 'Blur',
    ]


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Pet2Dataset:
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        # path = CFG.test_root + self.X[item] + '.jpg'
        path = self.df['image_path'].values[item]
        features = cv2.imread(path)
        features = cv2.cvtColor(features, cv2.COLOR_BGR2RGB)
        features = self.transforms['valid'](image=features)['image']
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return {
            'x': torch.tensor(features, dtype=torch.float32),
        }


class Pet2Model(nn.Module):
    def __init__(self, model_name):
        super(Pet2Model, self).__init__()    
        
        # Model Encoder
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0, in_chans=CFG.in_chans)
        self.model.head = nn.Linear(self.model.num_features, 128)
        self.dense = nn.Linear(128, CFG.TARGET_DIM)

    def forward(self, features):
        x = self.model(features)
        output = self.dense(x)
        return output.squeeze(-1)


class Pet2CNNModel(nn.Module):
    def __init__(self, model_name):
        super(Pet2CNNModel, self).__init__()    
        
        # Model Encoder
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0, in_chans=CFG.in_chans)
        self.model.fc = nn.Linear(self.model.num_features, 128)
        self.dense = nn.Linear(128, CFG.TARGET_DIM)

    def forward(self, features):
        x = self.model(features)
        output = self.dense(x)
        return output.squeeze(-1)


class Pet2VitCNNModel(nn.Module):
    def __init__(self, model_name):
        super(Pet2VitCNNModel, self).__init__()    
        
        # Model Encoder
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0, in_chans=CFG.in_chans)
        self.model.head = nn.Linear(self.model.num_features, 128)
        self.dense = nn.Linear(128, CFG.TARGET_DIM)

    def forward(self, features):
        x = self.model(features)
        output = self.dense(x)
        return output.squeeze(-1)


def make_preds(model_dict1, model_dict2, model_dict3, model_dict4, model_dict5):    
    # df = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
    df = pet1_image_all.copy()
    y_pred1 = []
    y_pred2 = []
    y_pred3 = []
    y_pred4 = []
    y_pred5 = []
    for fold in CFG.folds:
        model1 = Pet2Model(model_dict1['MODEL_NAME'])
        model1.to(device)
        model1.load_state_dict(torch.load(f'../input/kaerururu-petfinder2-{model_dict1["EXP_ID"]}/fold-{fold}.bin'))
        model1.eval()
        
        model2 = Pet2Model(model_dict2['MODEL_NAME'])
        model2.to(device)
        model2.load_state_dict(torch.load(f'../input/kaerururu-petfinder2-{model_dict2["EXP_ID"]}/fold-{fold}.bin'))
        model2.eval()
        
        model3 = Pet2Model(model_dict3['MODEL_NAME'])
        model3.to(device)
        model3.load_state_dict(torch.load(f'../input/kaerururu-petfinder2-{model_dict3["EXP_ID"]}/fold-{fold}.bin'))
        model3.eval()
        
        model4 = Pet2CNNModel(model_dict4['MODEL_NAME'])
        model4.to(device)
        model4.load_state_dict(torch.load(f'../input/kaerururu-petfinder2-{model_dict4["EXP_ID"]}/fold-{fold}.bin'))
        model4.eval()
        
        model5 = Pet2VitCNNModel(model_dict5['MODEL_NAME'])
        model5.to(device)
        model5.load_state_dict(torch.load(f'../input/kaerururu-petfinder2-{model_dict5["EXP_ID"]}/fold-{fold}.bin'))
        model5.eval()
        
        dataset1 = Pet2Dataset(df=df, transforms=model_dict1['TRANSFORM'])
        data_loader1 = torch.utils.data.DataLoader(
            dataset1, batch_size=CFG.valid_bs//4, num_workers=0, pin_memory=True, shuffle=False
        )
        
        dataset2 = Pet2Dataset(df=df, transforms=model_dict2['TRANSFORM'])
        data_loader2 = torch.utils.data.DataLoader(
            dataset2, batch_size=CFG.valid_bs//4, num_workers=0, pin_memory=True, shuffle=False
        )
        
        del dataset1, dataset2; gc.collect()

        final_output1 = []
        final_output2 = []
        final_output3 = []
        final_output4 = []
        final_output5 = []
        for b_idx, (data1, data2) in tqdm(enumerate(zip(data_loader1, data_loader2)), total=len(data_loader1)):
            with torch.no_grad():
                inputs384 = data1['x'].to(device)
                inputs224 = data2['x'].to(device)
                                          
                output1 = model1(inputs384)
                output2 = model2(inputs224)
                output3 = model3(inputs224)
                output4 = model4(inputs224)
                output5 = model5(inputs384)
                
                output1 = torch.sigmoid(output1) * 100.
                output1 = output1.detach().cpu().numpy().tolist()
                final_output1.extend(output1)
                
                output2 = torch.sigmoid(output2) * 100.
                output2 = output2.detach().cpu().numpy().tolist()
                final_output2.extend(output2)
                
                output3 = torch.sigmoid(output3) * 100.
                output3 = output3.detach().cpu().numpy().tolist()
                final_output3.extend(output3)
                
                output4 = torch.sigmoid(output4) * 100.
                output4 = output4.detach().cpu().numpy().tolist()
                final_output4.extend(output4)
                
                output5 = torch.sigmoid(output5) * 100.
                output5 = output5.detach().cpu().numpy().tolist()
                final_output5.extend(output5)
                
        y_pred1.append(np.array(final_output1))
        y_pred2.append(np.array(final_output2))
        y_pred3.append(np.array(final_output3))
        y_pred4.append(np.array(final_output4))
        y_pred5.append(np.array(final_output5))
        torch.cuda.empty_cache()
        
    del model1, model2, model3, model4, model5; gc.collect()
    del final_output1, final_output2, final_output3, final_output4, final_output5; gc.collect()
    return y_pred1, y_pred2, y_pred3, y_pred4, y_pred5


# environment
set_seed(CFG.seed)

model_dict1 = {
    'EXP_ID' : '042', 
    'MODEL_NAME' : 'swin_large_patch4_window12_384',
    'TRANSFORM' : {
        'valid' : A.Compose([
            A.Resize(384, 384, p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0,),
        ], p=1.0),    
    }
}

model_dict2 = {
    'EXP_ID' : '069', # '044', 
    'MODEL_NAME' : 'swin_base_patch4_window7_224_in22k', # 'swin_large_patch4_window7_224',
    'TRANSFORM' : {
        'valid' : A.Compose([
            A.Resize(224, 224, p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0,),
        ], p=1.0),    
    }
}

model_dict3 = {
    'EXP_ID' : '055', 
    'MODEL_NAME' : 'swin_large_patch4_window7_224',
    'TRANSFORM' : {
        'valid' : A.Compose([
            A.Resize(224, 224, p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0,),
        ], p=1.0),    
    }
}

model_dict4 = {
    'EXP_ID' : '060', 
    'MODEL_NAME' : 'resnext50_32x4d',
    'TRANSFORM' : {
        'valid' : A.Compose([
            A.Resize(224, 224, p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0,),
        ], p=1.0),    
    }
}

model_dict5 = {
    'EXP_ID' : '062', 
    'MODEL_NAME' : 'vit_base_r50_s16_384',
    'TRANSFORM' : {
        'valid' : A.Compose([
            A.Resize(384, 384, p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0,),
        ], p=1.0),    
    }
}


pet1_image_all = pd.DataFrame()
pet1_image_all['image_name'] = os.listdir('input/petfinder1_images/')
pet1_image_all['PetID'] = pet1_image_all['image_name'].map(lambda x: x.split('-')[0])
pet1_image_all['image_path'] = pet1_image_all['image_name'].map(lambda x: 'input/petfinder1_images/'+x)

print(pet1_image_all.shape)
print(pet1_image_all.head())


preds_042, preds_069, preds_055, preds_060, preds_062 = make_preds(model_dict1, model_dict2, model_dict3, model_dict4, model_dict5)


pseudo_label_0 = (preds_042[0] + preds_069[0] + preds_055[0] + preds_060[0] + preds_062[0])/5
pseudo_label_1 = (preds_042[1] + preds_069[1] + preds_055[1] + preds_060[1] + preds_062[1])/5
pseudo_label_2 = (preds_042[2] + preds_069[2] + preds_055[2] + preds_060[2] + preds_062[2])/5
pseudo_label_3 = (preds_042[3] + preds_069[3] + preds_055[3] + preds_060[3] + preds_062[3])/5
pseudo_label_4 = (preds_042[4] + preds_069[4] + preds_055[4] + preds_060[4] + preds_062[4])/5


pet1_image_all['pseudo_label'] = pseudo_label

print(pet1_image_all.shape)
print(pet1_image_all.head())


pet1_image_all.to_csv('input/petfinder1_train_test_image_with_pseudo_label.csv', index=False)


