import pandas as pd


def load_data():
    """
    Get price and hexagon id for each accommodation ID
    :return: pd.DataFrame
    """
    listings = pd.read_csv('../data/listings.csv')[['id', 'price']]
    hexagon_ids = pd.read_pickle('../data/hexagon_ids.pkl')
    return pd.merge(listings, hexagon_ids, left_on='id', right_index=True)


def target_encoding(data):
    """
    Hexagon ID는 Categorical variable로 대다수의 Algorithm에 적용하기 위해 numerical variables로 변환이 필요함
    OneHotEncoder가 갖는 Sparsity 문제를 피하기 위해 Mean encoding 방법을 적용
    Naive Mean encoding은 Information leakage문제를 내포하므로 Catboost의 Ordered Target Encoding의 Idea를 반영해 이를 해결함
    :param data: pd.DataFrame
    :return: Hexagon ID에 대한 encoded value
    """
    global_mean = data['price'].mean()
    encoded_value = []
    for i, [id, _, hexagon] in data.iterrows():
        value = (data.loc[data.hexagons == hexagon]
                    .loc[data.id != id, 'price']
                    .mean())
        encoded_value.append(value if value is not None else global_mean)
    return encoded_value


def main():
    data = load_data()
    encoded_value = target_encoding(data)
    result = pd.concat([data, pd.Series(encoded_value, name='neighborhood_value')], axis=1)
    result.to_pickle('./preprocessed/encoded_neighborhoods.pkl')


if __name__ == '__main__':
    main()