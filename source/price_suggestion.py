import numpy as np
import pickle
import pandas as pd


def load_proper_prices():
    lgbm_prediction = np.load('./result/lgbm_prediction.npy').reshape(-1, 1)
    rf_prediction = np.load('./result/rf_prediction.npy').reshape(-1, 1)
    nn_prediction = np.load('./result/nn_prediction.npy')
    return lgbm_prediction, rf_prediction, nn_prediction


def perform_ensemble(lgbm, rf, nn):
    concatenated = np.concatenate((lgbm, rf), axis=1)
    return np.average(concatenated, axis=1)


if __name__ == '__main__':
    a, b, _ = load_proper_prices()
    suggested_price = perform_ensemble(a, b, _)
    with open('./preprocessed/problematic_accommodations.pkl', 'rb') as f:
        accommodations_id = pickle.load(f)
    with open('./preprocessed/y_test.pkl', 'rb') as f:
        current_price = pickle.load(f)

    price_suggestion = pd.DataFrame({'suggested_price': suggested_price, 'current_price': current_price},
                                    index=accommodations_id)
    price_suggestion.to_csv('./result/price_suggestion.csv')
