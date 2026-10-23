# Bike Rental Prediction

An end-to-end machine learning project predicting daily bike rental counts from weather and calendar features. Built with pandas and scikit-learn while studying Chapter 2 of *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*.

## Dataset

[UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), daily file `day.csv`: 731 observations from 2011–2012. Target: `cnt` (daily rental count). Data is not bundled; consult UCI for attribution and license terms.

Numeric features: `temp`, `atemp`, `hum`, `windspeed`.
Categorical features: `season`, `yr`, `mnth`, `holiday`, `weekday`, `workingday`, `weathersit`.
`casual` and `registered` are excluded because their sum equals the target (target leakage). `instant` and the raw date are also excluded.

## Method

- Chronological 80/20 split: 584 training days, 147 held-out test days.
- Numeric imputation and standardization; categorical imputation and one-hot encoding.
- Full preprocessing/model pipelines evaluated using `TimeSeriesSplit(n_splits=5)`.
- Compare Linear Regression, Decision Tree, and Random Forest.
- Tune Random Forest with GridSearchCV (9 combinations).
- Choose the model using validation RMSE, then evaluate once on the held-out test set.

## Results

| Model | Training RMSE | Mean validation RMSE |
|---|---:|---:|
| Linear Regression | 678.94 | **1032.84** |
| Decision Tree | 0.00 | 1473.44 |
| Random Forest | 229.06 | 1297.29 |
| Tuned Random Forest | — | 1288.25 |

Best searched forest settings: `max_depth=5`, `min_samples_leaf=1`.
Selected model: **Linear Regression**. **Test RMSE: 1102.66 rentals/day**.
RMSE is not a percentage or mean absolute error. Results are from the original Colab run; outputs are retained, but this cleaned portable notebook has not been rerun in a fresh environment. Dependencies are not version-locked, so exact numerical reproduction is not guaranteed.

## Run

1. Download and extract `day.csv` from the UCI link above into `data/day.csv`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `bike_rental_prediction.ipynb` in Jupyter and run all cells in order.

For Colab: upload the notebook and upload `day.csv` to the runtime, then run all cells. No Google Drive mount is required. The save cell writes the fitted preprocessing/model pipeline to `models/bike_model.pkl`; download this file if you want to retain it after the Colab runtime ends. Model artifacts and data files are excluded by `.gitignore`.

## Limitations

The model follows the broad demand pattern but misses some peaks and sudden drops. Validation folds cover different time periods and training sizes. The dataset spans only two historical years, and the encoded `yr` feature does not represent a continuing future trend. For real future demand forecasts, weather forecasts would need to replace observed daily weather. No web service, production monitoring, or automated retraining has been implemented.

## Files

- `bike_rental_prediction.ipynb`: workflow, explanations, and saved outputs.
- `requirements.txt`: Python dependencies.
- `data/README.md`: dataset setup.
- `.gitignore`: excludes local data, models, and environment files.
