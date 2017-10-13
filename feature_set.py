import pandas as pd

train = pd.read_csv('data/train.csv')

with open('features/features.txt', 'r') as feat_file:
    best_features = [line.strip() for line in feat_file]
    feat_file.close()


cat_features = [feat for feat in train.columns if feat.split('_')[-1] == 'cat']
for feat in cat_features:
    try:
        dummy = pd.read_csv('data/%s.csv' % feat)
        train = train.join(dummy.set_index('id'), on='id', how='left', rsuffix='_onehot')
        train = train.drop(feat, axis=1)
        print("joined feature %s" % feat)
    except IOError:
        print("failed to open file data/%s.csv" % feat)


train = train[['id', 'target'] + best_features].set_index('id')
train.to_csv('data/p-train.csv')


