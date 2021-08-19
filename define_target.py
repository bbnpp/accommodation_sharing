import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture
from scipy.stats import mode

calendar = pd.read_csv('./data/calendar.csv')

calendar['date'] = pd.to_datetime(calendar.date)
period_condition = (calendar.date.dt.year == 2021) & (calendar.date.dt.month == 7)
subset = calendar.loc[period_condition]

subset.available = subset.available.map(lambda x: 1 if x=='t' else 0)
vacant_days = subset[['listing_id', 'available']].groupby('listing_id').sum()

clf = mixture.GaussianMixture(n_components=1, covariance_type='diag')
thresholds = 3.15
clf.fit(vacant_days)

target_idx = vacant_days.loc[-clf.score_samples(vacant_days) < 3.15].index.values

