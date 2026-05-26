# priceED

Marketplace pricing and sell-through modelling project.


## What it does

The project builds two machine learning models:

1. A classification model that predicts whether a listing will sell within 30 days.
2. A regression model that estimates days to sell for sold listings.

## Main outputs

For a given listing, the app returns:

- suggested listing price
- predicted 30-day sale probability
- estimated days to sell
- expected revenue
- pricing score

## Tech stack

- Python
- pandas
- scikit-learn
- LightGBM
- Plotly
- Streamlit

## Data

The dataset is synthetic.

It includes:

- product category
- condition
- brand type
- listing price
- market median price
- seller rating
- demand index
- views
- favourites
- messages
- sale outcome
- days to sell

## Model results

Sale probability model:

- Accuracy: 0.64
- Macro F1: 0.62
- Weighted F1: 0.64

Days-to-sell model:

- MAE: 4.09 days
- RMSE: 5.12 days
- R2: 0.7516