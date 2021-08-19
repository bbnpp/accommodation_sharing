import pandas as pd
import pickle


def load_dataset():
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
    dummy_room_types = pd.get_dummies(data['room_type'], prefix='type', drop_first=True)
    data = pd.merge(data.drop('room_type', axis=1), dummy_room_types, left_index=True, right_index=True)
    return data


def save_datasets(data):
    """
    Get X and y
    :param data: Preprocessed pd.DataFrame
    :return: np.array
    """
    X = data.drop('price', axis=1).fillna(0).values
    y = data['price'].values
    with open('./preprocessed/X.pkl', 'wb') as f:
        pickle.dump(X, f)
    with open('./preprocessed/y.pkl', 'wb') as f:
        pickle.dump(y, f)


def main():
    data = load_dataset()
    data = handle_categorical_variable(data)
    save_datasets(data)


if __name__ == '__main__':
    main()