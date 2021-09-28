import sys
sys.path.append("/root/workspace/PetFinderPawpularityContest")

import cv2
import os
import numpy as np
import pandas as pd

train = pd.read_csv('input/train.csv')


def get_train_file_path(image_id):
    return "input/train/{}.jpg".format(image_id)

train['file_path'] = train['Id'].apply(get_train_file_path)

OUT_DIR = 'input/train_npy/'

def save_images(file_path):
    file_name = file_path.split('/')[-1].split('.jpg')[0] + '.npy'
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    np.save(OUT_DIR + file_name, image)

import joblib
from tqdm.auto import tqdm

_ = joblib.Parallel(n_jobs=8)(
    joblib.delayed(save_images)(file_path) for file_path in tqdm(train['file_path'].values)
)

img1 = np.load('input/train_npy/0007de18844b0dbbb5e1f607da0606e0.npy')
print(img1.shape)

