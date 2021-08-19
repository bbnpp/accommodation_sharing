import h3 as h3
import pandas as pd


'''
listing 데이터로 할 수 있음
    accomodation 별 위경도 좌표를 확인한뒤
    accomodation 이 속하는 hexagon ID를 attach

calendar 데이터의 subset으로 할 수 있음
    hexagon별 평균가격을 구하는 데 정보유출 경계해서 뺄 것
'''


def load_listings():
    data = pd.read_csv('./data/listings.csv')
    return data['id'], data[['latitude', 'longitude']]


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





data = load_listings()
get_hexagons_from_coodinates(data[['latitude', 'longitude']].values)


def main():
    print(listings.shape)


if __name__ == '__main__':
    main()

