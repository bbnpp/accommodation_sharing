import pickle
import pandas as pd
from sklearn import mixture


def load_reservation_history():
    """
    calendar data로부터 일부 기간('2021.07)을 Slicing
    :return pd.DataFrame: Sliced history data
    """
    calendar = pd.read_csv('./data/calendar.csv')
    calendar['date'] = pd.to_datetime(calendar.date)
    period_condition = (calendar.date.dt.year == 2021) & (calendar.date.dt.month == 7)
    subset = calendar.loc[period_condition]
    return subset


def get_vacant_days(subset):
    """
    Get the number of vacant days for each accomodation in a specific month
    :return pd.DataFrame: vacant_days
    """
    subset.available = subset.available.map(lambda x: 1 if x=='t' else 0)
    vacant_days = subset[['listing_id', 'available']].groupby('listing_id').sum()
    return vacant_days


def get_problematic_accommodations(vacant_days, threshold):
    """
    Fit gaussian mixture distributions to the vacant_days data
    Define problematic accommodations which are too popular or unpopular ones
    :param vacant_days: vacant days for each accommodation
    :param threshold: threshold for define outliers of a distribution
    :return: ID of problematic accommodations
    """
    clf = mixture.GaussianMixture(n_components=3, covariance_type='diag')
    clf.fit(vacant_days)
    condition = abs(clf.score_samples(vacant_days)) >= threshold
    problematic_id = vacant_days.loc[condition].index.values
    return problematic_id


def main():
    subset = load_reservation_history()
    vacant_days = get_vacant_days(subset)
    problematic_id = get_problematic_accommodations(vacant_days, 5.5)
    with open('./preprocessed/problematic_accommodations.pkl', 'wb') as f:
        pickle.dump(problematic_id, f)


if __name__ == '__main__':
    main()

