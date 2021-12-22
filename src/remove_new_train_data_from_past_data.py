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
import timm


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
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


pet1_data = pd.read_csv('input/petfinder1_train_test_image_with_pseudo_label.csv')
print(pet1_data.shape)
print(pet1_data.head())

sim_data = pd.read_csv('input/pawpularity_adoptionSpeed.csv', usecols=['org_id', 'pre_id', 'sim'])
sim_data = sim_data[sim_data['sim']>0.6].reset_index(drop=True)
print(sim_data['pre_id'].nunique())
print(sim_data.shape)
print(sim_data.head())

pet1_data = pet1_data[~pet1_data['PetID'].isin(sim_data['pre_id'])].reset_index(drop=True)
print(pet1_data.shape)
print(pet1_data.head())

sim_eye_grep = pd.read_csv('input/same_img_pair_check_all.csv')
print(sim_eye_grep['PetID'].nunique())
print(sim_eye_grep.shape)
print(sim_eye_grep.head())

pet1_data = pet1_data[~pet1_data['PetID'].isin(sim_eye_grep['PetID'])].reset_index(drop=True)
print(pet1_data.shape)
print(pet1_data.head())

pet1_data.to_csv('input/petfinder1_train_test_image_with_pseudo_label_rm_new_train_data_more_clean.csv', index=False)


