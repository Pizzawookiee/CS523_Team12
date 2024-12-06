import pandas as pd
import numpy as np
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_error

original_folder = 'C:/Users/damol/OneDrive/Desktop/Verify'
train_file = os.path.join(original_folder, 'train_data.parquet')
test_file = os.path.join(original_folder, 'test_data.parquet')

train_data = pd.read_parquet(train_file)
test_data = pd.read_parquet(test_file)


train_data['datetime'] = pd.to_datetime(
    train_data['ID'].str.split('_').str[1] + ' ' + train_data['ID'].str.split('_').str[2], format='%y-%m-%d %H%M'
)

test_data['datetime'] = pd.to_datetime(
    test_data['ID'].str.split('_').str[1] + ' ' + test_data['ID'].str.split('_').str[2], format='%y-%m-%d %H%M'
)
# This Create lag features for target variable 'Value' (next 3 hours = 12 intervals)
train_data['Value_lag_1'] = train_data['Value'].shift(-1)
train_data['Value_lag_2'] = train_data['Value'].shift(-2)
train_data['Value_lag_3'] = train_data['Value'].shift(-3)
train_data['Value_lag_4'] = train_data['Value'].shift(-4)
train_data['Value_lag_5'] = train_data['Value'].shift(-5)
train_data['Value_lag_6'] = train_data['Value'].shift(-6)
train_data['Value_lag_7'] = train_data['Value'].shift(-7)
train_data['Value_lag_8'] = train_data['Value'].shift(-8)
train_data['Value_lag_9'] = train_data['Value'].shift(-9)
train_data['Value_lag_10'] = train_data['Value'].shift(-10)
train_data['Value_lag_11'] = train_data['Value'].shift(-11)
train_data['Value_lag_12'] = train_data['Value'].shift(-12)

#This Drop rows with NaN values resulting from lag features
train_data = train_data.dropna(subset=['Value_lag_1', 'Value_lag_2', 'Value_lag_3', 'Value_lag_4', 
                                       'Value_lag_5', 'Value_lag_6', 'Value_lag_7', 'Value_lag_8',
                                       'Value_lag_9', 'Value_lag_10', 'Value_lag_11', 'Value_lag_12'])

#Features and target for training
features = ['temperature', 'visibility', 'wind_speed', 'cloud_severity', 
            'Value_lag_1', 'Value_lag_2', 'Value_lag_3', 'Value_lag_4', 
            'Value_lag_5', 'Value_lag_6', 'Value_lag_7', 'Value_lag_8',
            'Value_lag_9', 'Value_lag_10', 'Value_lag_11', 'Value_lag_12']
target = 'Value'

X_train = train_data[features]
y_train = train_data[target]


preprocessor = StandardScaler()


model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(random_state=42))
])


param_dist = {
    'regressor__n_estimators': randint(100, 1000),
    'regressor__learning_rate': uniform(0.01, 0.1),
    'regressor__max_depth': randint(3, 10),
    'regressor__subsample': uniform(0.6, 0.4),
    'regressor__colsample_bytree': uniform(0.6, 0.4),
    'regressor__num_leaves': randint(31, 200)
}


random_search = RandomizedSearchCV(
    model_pipeline,
    param_distributions=param_dist,
    n_iter=30,
    scoring='neg_mean_squared_error',
    cv=3,
    random_state=42,
    verbose=1,
    n_jobs=-1
)


random_search.fit(X_train, y_train)


best_model = random_search.best_estimator_


y_train_pred = best_model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
score_train = np.exp(-rmse_train / 10)

print(f"Validation RMSE: {rmse_train:.4f}")
print(f"Validation Score (exp(-RMSE/10)): {score_train:.4f}")


test_data['datetime'] = pd.to_datetime(
    test_data['ID'].str.split('_').str[1] + ' ' + test_data['ID'].str.split('_').str[2], format='%y-%m-%d %H%M'
)

# Generate lag features for test data (only use past data for prediction)
test_data['Value_lag_1'] = test_data['Value'].shift(-1)
test_data['Value_lag_2'] = test_data['Value'].shift(-2)
test_data['Value_lag_3'] = test_data['Value'].shift(-3)
test_data['Value_lag_4'] = test_data['Value'].shift(-4)
test_data['Value_lag_5'] = test_data['Value'].shift(-5)
test_data['Value_lag_6'] = test_data['Value'].shift(-6)
test_data['Value_lag_7'] = test_data['Value'].shift(-7)
test_data['Value_lag_8'] = test_data['Value'].shift(-8)
test_data['Value_lag_9'] = test_data['Value'].shift(-9)
test_data['Value_lag_10'] = test_data['Value'].shift(-10)
test_data['Value_lag_11'] = test_data['Value'].shift(-11)
test_data['Value_lag_12'] = test_data['Value'].shift(-12)

#Drop rows with NaN values resulting from lag features in test data
test_data = test_data.dropna(subset=['Value_lag_1', 'Value_lag_2', 'Value_lag_3', 'Value_lag_4', 
                                     'Value_lag_5', 'Value_lag_6', 'Value_lag_7', 'Value_lag_8',
                                     'Value_lag_9', 'Value_lag_10', 'Value_lag_11', 'Value_lag_12'])


X_test = test_data[features]


y_test_pred = best_model.predict(X_test)

test_ids = test_data['ID']
submission_df = pd.DataFrame({'ID': test_ids, 'Value': y_test_pred})
submission_df = submission_df.sort_values('ID', ascending=True)


submission_file = os.path.join(original_folder, 'prediction.csv')
submission_df.to_csv(submission_file, index=False)
print(f"Submission saved to {submission_file}")


print("Sample of submission:")
print(submission_df.head())

print(f"Validation Score (exp(-RMSE/10)): {score_train:.4f}")
