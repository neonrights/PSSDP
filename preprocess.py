import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
train.replace(-1, np.nan)

# preprocess categorical data, give suitable names, handle missing data/-1
for feature in train.columns:
    if feature.split('_')[-1] == 'cat':
        # check number of categories
        n = train[feature].nunique()
        if n < 30:
            # one hot encode categorical data
            dummy = pd.get_dummies(train[feature], prefix=feature)
            dummy['id'] = train['id']
            dummy.to_csv("data/%s.csv" % feature, mode='w+')
            print("saved one hot encoded %s to %s.csv" % (feature, feature))
        else:
            print("skipped %s, %d categories" % (feature, n))


