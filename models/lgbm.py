import numpy as np
import lightgbm as lgbm
from sklearn.model_selection import RandomizedSearchCV
from source import build_dataset


def define_model():
    regressor = lgbm.LGBMRegressor()
    return regressor


def tune_model(regressor, params):
    tuner = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=params,
        n_iter=10,
        cv=3,
        refit=True,
        verbose=True)

    tuner.fit(X_train, y_train)
    print(f'Best mse: {tuner.best_score_:.4f}')
    return tuner


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = build_dataset.train_test_split(fp='./preprocessed')
    params = {'max_depth': [-1, 5, 10, 15],
              'learning_rate': [0.1, 0.01, 0.001, 0.0001],
              'subsample': [0.6, 0.8, 1.0]
              }

   lgbm = define_model()
   tuned_lgbm = tune_model(lgbm, params)

    with open('./result/lgbm_prediction.npy', 'wb') as f:
        np.save(f, tuned_lgbm.predict(X_test))