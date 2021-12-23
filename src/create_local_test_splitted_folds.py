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
# ↓ higher ids
not_use_dup_ids = [
 '373c763f5218610e9b3f82b12ada8ae5',
 '67e97de8ec7ddcda59a58b027263cdcc',
 '839087a28fa67bf97cdcaf4c8db458ef',
 '264845a4236bc9b95123dde3fb809a88',
 'def7b2f2685468751f711cc63611e65b',
 '3f0222f5310e4184a60a7030da8dc84b',
 'c504568822c53675a4f425c8e5800a36',
 '86a71a412f662212fe8dcd40fdaee8e6',
 'a8bb509cd1bd09b27ff5343e3f36bf9e',
 '0b04f9560a1f429b7c48e049bcaffcca',
 '68e55574e523cf1cdc17b60ce6cc2f60',
 'bca6811ee0a78bdcc41b659624608125',
 '5da97b511389a1b62ef7a55b0a19a532',
 'c25384f6d93ca6b802925da84dfa453e',
 '08440f8c2c040cf2941687de6dc5462f',
 '0c4d454d8f09c90c655bd0e2af6eb2e5',
 '5a5c229e1340c0da7798b26edf86d180',
 '871bb3cbdf48bd3bfd5a6779e752613e',
 'dbf25ce0b2a5d3cb43af95b2bd855718',
 '43bd09ca68b3bcdc2b0c549fd309d1ba',
 '43ab682adde9c14adb7c05435e5f2e0e',
 'b86589c3e85f784a5278e377b726a4d4',
 '6cb18e0936faa730077732a25c3dfb94',
 '589286d5bfdc1b26ad0bf7d4b7f74816',
 'b967656eb7e648a524ca4ffbbc172c06',
 'e09a818b7534422fb4c688f12566e38f',
 '902786862cbae94e890a090e5700298b',
 '8f20c67f8b1230d1488138e2adbb0e64',
 '221b2b852e65fe407ad5fd2c8e9965ef',
 '41c85c2c974cc15ca77f5ababb652f84',
 '6dc1ae625a3bfb50571efedc0afc297c',
 '763d66b9cf01069602a968e573feb334',
 '03d82e64d1b4d99f457259f03ebe604d',
 '851c7427071afd2eaf38af0def360987',
 'b956edfd0677dd6d95de6cb29a85db9c',
 'd050e78384bd8b20e7291b3efedf6a5b',
 '04201c5191c3b980ae307b20113c8853'
]


# read training data
df = pd.read_csv("input/train.csv")

df = df[~df['Id'].isin(not_use_dup_ids)].reset_index(drop=True)

# create folds
NUM_SPLITS = 5
df = create_folds(df, num_splits=NUM_SPLITS)

oof = pd.read_csv("output/055/oof.csv")
oof['diff'] = oof['Pawpularity'] - oof['oof']
oof = oof.sort_values('diff', ascending=False)
hard_index = oof.head(1800).index

df.loc[hard_index, 'kfold'] = -1

print(df.kfold.value_counts())

# df.to_csv(f"input/train_folds_{NUM_SPLITS}.csv", index=False)
df.to_csv(f"input/train_folds_no_dup_5_sep_hard_local_test.csv", index=False)

