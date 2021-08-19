import h3 as h3
import numpy as np
import pandas as pd


def load_listings():
    data = pd.read_csv('./data/listings.csv')
    return data['id'], data[['latitude', 'longitude']].values


def get_hexagons_from_coodinates(coordinates: np.array, resolution: int = 8):
    """
    WGS84를 따르는 위경도 좌표를 arguments로 받아 해당하는 HexagonID를 반환
    :param coordinates: WGS84 위경도좌표(lat, lon)
    :param resolution: Hexagon ResolutionE
    :return: Hexagon ids
    """
    hexagons = []
    for x, y in coordinates:
        hexagon_id = h3.geo_to_h3(x, y, resolution=resolution)
        hexagons.append(hexagon_id)
    return np.array(hexagons)


def attach_hexagon_id(id, hexagons):
    """
    accommodation id 별 Hexagon id를 mapping하여 pd.Series로 return
    :param id: accommodation ids
    :param hexagons: hexagon id for each accommodation
    :return: pd.Series
    """
    return pd.Series(hexagons, index=id, name='hexagons')


def main():
    id, coordinates = load_listings()
    hexagons = get_hexagons_from_coodinates(coordinates)
    result = attach_hexagon_id(id, hexagons)
    result.to_pickle('./preprocessed/hexagon_ids.pkl')


if __name__ == '__main__':
    main()
