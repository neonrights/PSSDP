import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

FOLDER = 'features'

# lead data set
train = pd.read_csv('data/train.csv')

# merge one hot features
cat_features = [feat for feat in train.columns if feat.split('_')[-1] == 'cat']
for feat in cat_features:
    try:
        dummy = pd.read_csv('data/%s.csv' % feat)
        train = train.join(dummy.set_index('id'), on='id', how='left', rsuffix='_onehot')
        train = train.drop(feat, axis=1)
    except IOError:
        print("failed to open file data/%s.csv" % feat)

# remove id's and target variables
features = train[[feat for feat in train.columns if not (feat.startswith('id') or feat == 'target')]]
targets = train.target

# get ten best features for random forest
estimator = RandomForestClassifier(max_depth=3)
selector = RFE(estimator, 50, verbose=1)
selector.fit(features, targets)

# make histograms of ten best features
best_features = features.columns[selector.support_]
good_drivers = train[train.target == 0]
bad_drivers = train[train.target == 1]

# clear folder
for f in os.listdir(FOLDER):
    os.remove(os.path.join(FOLDER, f))

# record best features
featfile = open("%s/features.txt" % FOLDER, 'w+')

for feature in best_features:
    featfile.write("%s\n" % feature)

    # set bins for category type
    cat = feature.split('_')
    if cat[-1] == 'cat':
        bins = train[feature].nunique()
    elif cat[-1] == 'bin' or cat[-2] == 'cat':
        bins = 2
    else:
        bins = 30

    # remove -1 aka missing values
    x = good_drivers[feature]
    y = bad_drivers[feature]

    x = x[x != -1]
    y = y[y != -1]
    
    # create histogram for visual pruning
    fig = plt.figure()

    plt.hist(x.values, bins=bins, alpha=0.5, label='good drivers', normed=True)
    plt.hist(y.values, bins=bins, alpha=0.5, label='bad drivers', normed=True)

    plt.legend(loc='upper right')
    plt.title(feature)
    
    fig.savefig('%s/%s.png' % (FOLDER, feature))
    plt.close(fig)

print(list(best_features))


