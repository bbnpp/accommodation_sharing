#!/bin/python

read pass
  python3 ./source/hexagon.py "$pass"
  python3 ./source/mean_encoder.py "$pass"
  python3 ./source/get_outliers.py "$pass"
  python3 ./source/build_dataset.py "$pass"
  python3 ./models/lgbm.py "$pass"
  python3 ./models/random_forest.py "$pass"
  python3 ./models/neural_network.py "$pass"
  python3 ./source/price_suggestion.py "$pass"

