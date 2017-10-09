import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
good = train[train.target == 0]
bad = train[train.target == 1]

for feature in train.columns:
    if feature.split('_')[-1] == 'cat':
        temp = train[feature]
        bins = len(set(temp[temp != -1]))

        x = good[feature]
        y = bad[feature]

        x = x[x != -1]
        y = y[y != -1]

        fig = plt.figure()

        plt.hist(x, bins=bins, alpha=0.5, label='good drivers', normed=True)
        plt.hist(y, bins=bins, alpha=0.5, label='bad drivers', normed=True)
         
        plt.legend(loc='upper right')
        plt.title(feature)
             
        fig.savefig('%s.png' % feature)
        print("saved %s.png" % feature)
        
