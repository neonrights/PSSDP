import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# lead data set
train = pd.read_csv('train.csv')
features = train[train.columns[2:]]
targets = train.target

# get ten best features for random forest
estimator = RandomForestClassifier(max_depth=3)
selector = RFE(estimator, 14, verbose=1)
selector.fit(features, targets)

# make histograms of ten best features
best_features = features.columns[selector.support_]
good_drivers = train[train.target == 0]
bad_drivers = train[train.target == 1]

# record best features
featfile = open("features.txt", 'w+')

for feature in best_features:
    featfile.write("%s\n" % feature)

    # set bins for category type
    cat = feature.split('_')[-1]
    if cat == 'cat':
        bins = len(set(train[feature])) - 1
    elif cat == 'bin':
        bins = 2
    else:
        bins = 50

    # remove -1 aka missing values
    x = good_drivers[feature]
    y = bad_drivers[feature]

    x = x[x != -1]
    y = y[y != -1]
    
    fig = plt.figure()

    plt.hist(x, bins=bins, alpha=0.5, label='good drivers', normed=True)
    plt.hist(y, bins=bins, alpha=0.5, label='bad drivers', normed=True)

    plt.legend(loc='upper right')
    plt.title(feature)
    
    fig.savefig('%s.png' % feature)

print(list(best_features))


