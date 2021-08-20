import pandas as pd
import pickle


def load_files():
    """
    Load files to build a dataset for training with manual feature selection
    :return: pd.DataFrame
    """
    listings = pd.read_csv('./data/listings.csv')[['id', 'room_type', 'minimum_nights', 'number_of_reviews',
                                                   'reviews_per_month', 'calculated_host_listings_count', 'price']]
    encoded_neighborhoods = pd.read_pickle('./data/encoded_neighborhoods.pkl')[['id', 'neighborhood_value']]
    data = pd.merge(listings, encoded_neighborhoods, left_on='id', right_on='id').fillna(0)
    return data.set_index('id')


def handle_categorical_variable(data):
    """
    As room_type feature is a categorical variable with low cardinality,
    transforming it to dummy variables might not cause a serious complexity problem.
    :return: pd.DataFrame
    """
    dummy_room_types = pd.get_dummies(data['room_type'], prefix='type', drop_first=False)
    data = pd.merge(data.drop('room_type', axis=1), dummy_room_types, left_index=True, right_index=True)
    return data


def get_targets(data):
    """
    Get X and y
    :param data: Preprocessed pd.DataFrame
    :return: pd.DataFrame
    """
    X = data.drop('price', axis=1).fillna(0)
    y = data['price']
    return X, y


def dump_datasets(X, y):
    target_idx = pd.read_pickle('./preprocessed/problematic_accommodations.pkl')
    X_train, X_test = X.drop(target_idx), X.loc[target_idx]
    y_train, y_test = y.drop(target_idx), y.loc[target_idx]

    with open('./preprocessed/X_train.pkl', 'wb') as f:
        pickle.dump(X_train.values, f)
    with open('./preprocessed/X_test.pkl', 'wb') as f:
        pickle.dump(X_test.values, f)
    with open('./preprocessed/y_train.pkl', 'wb') as f:
        pickle.dump(y_train.values, f)
    with open('./preprocessed/y_test.pkl', 'wb') as f:
        pickle.dump(y_test.values, f)


def train_test_split(fp):
    with open(f'{fp}/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open(f'{fp}/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open(f'{fp}/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open(f'{fp}/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    return X_train, X_test, y_train, y_test


def main():
    data = load_files()
    data = handle_categorical_variable(data)
    X, y = get_targets(data)
    dump_datasets(X, y)


if __name__ == '__main__':
    main()