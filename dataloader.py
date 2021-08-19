import pandas as pd

listings = pd.read_csv('./data/listings.csv') #key: id
calendar = pd.read_csv('./data/calendar.csv') #key: listing_id
neighborhood = pd.read_csv('./data/neighbourhoods.csv')
reviews = pd.read_csv('./data/reviews.csv')
summary_listing = pd.read_csv('./data/summary_listings.csv')
summary_review = pd.read_csv('./data/summary_reviews.csv')