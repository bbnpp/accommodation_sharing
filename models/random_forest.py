import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from source import build_dataset


def define_model():
    regressor = RandomForestRegressor(bootstrap=True)
    return regressor


def tune_model(regressor, params):
    tuner = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=params,
        n_iter=5,
        cv=3,
        refit=True,
        verbose=True)

    tuner.fit(X_train, y_train)
    print(f'Best mse: {tuner.best_score_:.4f}')
    return tuner


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = build_dataset.train_test_split(fp='./preprocessed')
    params = {'n_estimators': [n for n in range(100, 1_000, 200)],
              'max_features': ['auto', 'sqrt'],
              'max_depth': [n for n in range(5, 50, 10)],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]
              }

   rf = define_model()
   tuned_rf = tune_model(rf, params)

    with open('./result/rf_prediction.npy', 'wb') as f:
        np.save(f, tuned_rf.predict(X_test))
