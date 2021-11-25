import sys
sys.path.append("/root/workspace/PetFinderPawpularityContest")

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection


def create_folds(data, num_splits):
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    # data = data.sample(frac=1).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = int(np.floor(1 + np.log2(len(data))))
    
    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data['Pawpularity'], bins=num_bins, labels=False
    )
    
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    
    # drop the bins column
    data = data.drop("bins", axis=1)

    # return dataframe with folds
    return data


# If there is duplicate images, I'll use lower Pawpularity ones.
# https://www.kaggle.com/kaerunantoka/petfinder2-dup-analysis?scriptVersionId=80754319
use_dup_ids = ['13d215b4c71c3dc603cd13fc3ec80181',
 '5ef7ba98fc97917aec56ded5d5c2b099',
 'a8f044478dba8040cc410e3ec7514da1',
 '1feb99c2a4cac3f3c4f8a4510421d6f5',
 '3c50a7050df30197e47865d08762f041',
 '37ae1a5164cd9ab4007427b08ea2c5a3',
 '5a642ecc14e9c57a05b8e010414011f2',
 '2a8409a5f82061e823d06e913dee591c',
 '3c602cbcb19db7a0998e1411082c487d',
 '0422cd506773b78a6f19416c98952407',
 '9b3267c1652691240d78b7b3d072baf3',
 '1059231cf2948216fcc2ac6afb4f8db8',
 '8ffde3ae7ab3726cff7ca28697687a42',
 '78a02b3cb6ed38b2772215c0c0a7f78e',
 'bf8501acaeeedc2a421bac3d9af58bb7',
 'fe47539e989df047507eaa60a16bc3fd',
 'dd042410dc7f02e648162d7764b50900',
 '988b31dd48a1bc867dbc9e14d21b05f6',
 'e359704524fa26d6a3dcd8bfeeaedd2e',
 '6ae42b731c00756ddd291fa615c822a1',
 '9a0238499efb15551f06ad583a6fa951',
 'a9513f7f0c93e179b87c01be847b3e4c',
 '38426ba3cbf5484555f2b5e9504a6b03',
 'cd909abf8f425d7e646eebe4d3bf4769',
 '9f5a457ce7e22eecd0992f4ea17b6107',
 'b148cbea87c3dcc65a05b15f78910715',
 '3877f2981e502fe1812af38d4f511fd2',
 'b190f25b33bd52a8aae8fd81bd069888',
 '94c823294d542af6e660423f0348bf31',
 '2b737750362ef6b31068c4a4194909ed',
 '01430d6ae02e79774b651175edd40842',
 '72b33c9c368d86648b756143ab19baeb',
 'dbc47155644aeb3edd1bd39dba9b6953',
 'b49ad3aac4296376d7520445a27726de',
 '54563ff51aa70ea8c6a9325c15f55399',
 '87c6a8f85af93b84594a36f8ffd5d6b8',
 '16d8e12207ede187e65ab45d7def117b']


# read training data
df = pd.read_csv("input/train.csv")

df = df[df['Id'].isin(use_dup_ids)].reset_index(drop=True)

# create folds
NUM_SPLITS = 5
df = create_folds(df, num_splits=NUM_SPLITS)

print(df.kfold.value_counts())

# df.to_csv(f"input/train_folds_{NUM_SPLITS}.csv", index=False)
df.to_csv(f"input/train_folds_no_dup_{NUM_SPLITS}.csv", index=False)

