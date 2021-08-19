from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold


rf = RandomForestRegressor()

model = lgb.LGBMClassifier(n_estimators=100)


params = {'classifier__learning_rate': [0.01, 0.05, 0.1],
          'classifier__num_leaves': [3, 6, 9],
          'classifier__reg_alpha': [1e-1, 1, 10]
         }



tuner = RandomizedSearchCV(
    estimator=clf,
    param_distributions=params,
    n_iter=3,
    scoring='roc_auc',
    cv=3,
    refit=True,
    verbose=True,
    n_jobs=-1
)

tuner.fit(X, y)


print(f'Best auroc: {tuner.best_score_:.4f}')
