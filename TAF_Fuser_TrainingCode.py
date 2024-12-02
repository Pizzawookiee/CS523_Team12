import os
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import joblib
import pyarrow.parquet as pq
from xgboost.callback import EarlyStopping
# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# This are my  Data Directories 
# Define directories (update these paths as needed)
fuser_train_dir = r'C:\Users\damol\OneDrive\Desktop\Competition\Train_Set\Fuser_Train'
fuser_test_dir = r'C:\Users\damol\OneDrive\Desktop\Competition\Test_Set\Fuser_Test'
taf_train_dir = r'C:\Users\damol\OneDrive\Desktop\Competition\Train_Set\TAF_train_parquet'
taf_test_dir = r'C:\Users\damol\OneDrive\Desktop\Competition\Test_Set\TAF_test_parquet'

#Directories to save processed data
processed_data_dir = 'processed_data'
os.makedirs(processed_data_dir, exist_ok=True)

#My Helper Functions

def process_and_save_parquet_files_pyarrow(input_dir, output_dir, preprocess_func, file_prefix, columns_to_read=None):
    os.makedirs(output_dir, exist_ok=True)
    for file in tqdm(os.listdir(input_dir), desc=f"Processing files in {input_dir}"):
        if file.endswith('.parquet'):
            file_path = os.path.join(input_dir, file)
            output_file = os.path.join(output_dir, f"{file_prefix}_{file}")
            #I Removed  existing output file to ensure fresh processing
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except Exception as e:
                    print(f"Unable to delete existing file {output_file}: {e}")
                    continue
            try:
                #Read the Parquet file with PyArrow in batches
                parquet_file = pq.ParquetFile(file_path)
                dfs = []
                for batch in parquet_file.iter_batches(columns=columns_to_read, batch_size=1000000):
                    chunk = batch.to_pandas()
                    # Preprocess the data
                    chunk = preprocess_func(chunk)
                    if not chunk.empty:
                        dfs.append(chunk)
                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
                    # Save the processed data if it's not empty
                    df.to_parquet(
                        output_file,
                        index=False,
                        engine='pyarrow',
                        compression='snappy'
                    )
                    del df
                else:
                    print(f"No data to save for {file_path} after preprocessing.")
                del dfs
                gc.collect()
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")

def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object and col != 'station_id':
            df[col] = df[col].astype('category')
        elif col_type == 'int64':
            df[col] = df[col].astype('int32')
        elif col_type == 'float64':
            df[col] = df[col].astype('float32')
    return df

# Here is the Data Preprocessing Functions 

def preprocess_fuser(data):
    necessary_columns = [
        'arrival_runway_actual_time', 'departure_runway_actual_time',
        'estimated_arrival_time_prediction_timestamp', 'estimated_arrival_time',
        'estimated_departure_time_prediction_timestamp', 'estimated_departure_time',
        'is_arrival', 'arrival_airport'
    ]
    existing_columns = [col for col in necessary_columns if col in data.columns]
    if not existing_columns:
        return pd.DataFrame()
    data = data[existing_columns]
    datetime_cols = [col for col in existing_columns if 'time' in col]
    for col in datetime_cols:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    if 'is_arrival' in data.columns:
        data = data[data['is_arrival'] == True]
    data = data.drop_duplicates()
    if 'estimated_arrival_time' in data.columns and data['estimated_arrival_time'].notnull().any():
        data['time_bucket'] = data['estimated_arrival_time'].dt.floor('15min')
    elif 'arrival_runway_actual_time' in data.columns and data['arrival_runway_actual_time'].notnull().any():
        data['time_bucket'] = data['arrival_runway_actual_time'].dt.floor('15min')
    else:
        data['time_bucket'] = None
    if 'arrival_airport' in data.columns:
        data['arrival_airport'] = data['arrival_airport'].astype(str)
        data = data.dropna(subset=['arrival_airport'])
    data = data.dropna(subset=['time_bucket'])
    data = reduce_memory_usage(data)
    return data

def preprocess_taf(data):
    necessary_columns = ['datetime', 'station_id', 'temperature', 'visibility', 'wind_speed']
    existing_columns = [col for col in necessary_columns if col in data.columns]
    if not existing_columns:
        return pd.DataFrame()
    data = data[existing_columns]
    data['station_id'] = data['station_id'].astype(str)
    valid_station_ids = data['station_id'].str.fullmatch(r'.+')
    data = data[valid_station_ids]
    data = data.dropna(subset=['station_id'])
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        data = data.dropna(subset=['datetime'])
        data['time_bucket'] = data['datetime'].dt.floor('15min')
    else:
        data['time_bucket'] = None
    data = data.dropna(subset=['time_bucket'])
    data['time_bucket'] = data['time_bucket'].astype('datetime64[ns]')
    column_mapping = {
        'temperature': 'forecast_temperature',
        'visibility': 'forecast_visibility',
        'wind_speed': 'forecast_wind_speed'
    }
    data = data.rename(columns=column_mapping)
    data = reduce_memory_usage(data)
    return data

#Processing and Saving Data 

#Define columns to read for each data type
fuser_columns = [
    'arrival_runway_actual_time', 'departure_runway_actual_time',
    'estimated_arrival_time_prediction_timestamp', 'estimated_arrival_time',
    'estimated_departure_time_prediction_timestamp', 'estimated_departure_time',
    'is_arrival', 'arrival_airport'
]
taf_columns = ['datetime', 'station_id', 'temperature', 'visibility', 'wind_speed']

print("Processing and saving FUSER train data...")
process_and_save_parquet_files_pyarrow(
    input_dir=fuser_train_dir,
    output_dir=os.path.join(processed_data_dir, 'fuser_train'),
    preprocess_func=preprocess_fuser,
    file_prefix='fuser_train',
    columns_to_read=fuser_columns
)

print("Processing and saving FUSER test data...")
process_and_save_parquet_files_pyarrow(
    input_dir=fuser_test_dir,
    output_dir=os.path.join(processed_data_dir, 'fuser_test'),
    preprocess_func=preprocess_fuser,
    file_prefix='fuser_test',
    columns_to_read=fuser_columns
)

print("Processing and saving TAF train data...")
process_and_save_parquet_files_pyarrow(
    input_dir=taf_train_dir,
    output_dir=os.path.join(processed_data_dir, 'taf_train'),
    preprocess_func=preprocess_taf,
    file_prefix='taf_train',
    columns_to_read=taf_columns
)

print("Processing and saving TAF test data...")
process_and_save_parquet_files_pyarrow(
    input_dir=taf_test_dir,
    output_dir=os.path.join(processed_data_dir, 'taf_test'),
    preprocess_func=preprocess_taf,
    file_prefix='taf_test',
    columns_to_read=taf_columns
)

#Merging and Processing the dqata

def process_and_merge_data_pandas(fuser_files, taf_files, output_file_prefix):
    for idx, fuser_file in enumerate(tqdm(fuser_files, desc="Processing FUSER files")):
        output_file = f"{output_file_prefix}_{idx}.parquet"
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except Exception as e:
                continue
        fuser_data = pd.read_parquet(fuser_file)
        if fuser_data.empty:
            continue
        throughput = compute_throughput_pandas(fuser_data)
        fuser_data = fuser_data.merge(throughput, on=['arrival_airport', 'time_bucket'], how='left')
        airports = fuser_data['arrival_airport'].unique().tolist()
        time_buckets = fuser_data['time_bucket'].unique().tolist()
        taf_data = load_relevant_taf_data_pandas(taf_files, airports, time_buckets)
        merged_data = merge_datasets_pandas(fuser_data, taf_data)
        merged_data = feature_engineering_pandas(merged_data)
        if not merged_data.empty:
            merged_data.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')
        del fuser_data, taf_data, merged_data
        gc.collect()

def load_relevant_taf_data_pandas(taf_files, airports, time_buckets):
    relevant_taf = []
    for file in taf_files:
        if file.endswith('.parquet'):
            file_path = file
            if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
                continue
            taf_data = pd.read_parquet(file_path)
            if 'station_id' not in taf_data.columns or 'time_bucket' not in taf_data.columns:
                continue
            taf_data = taf_data[taf_data['station_id'].isin(airports)]
            taf_data = taf_data[taf_data['time_bucket'].isin(time_buckets)]
            if not taf_data.empty:
                relevant_taf.append(taf_data)
            del taf_data
            gc.collect()
    if relevant_taf:
        taf_data = pd.concat(relevant_taf, ignore_index=True)
        del relevant_taf
        gc.collect()
        return taf_data
    else:
        return None

def compute_throughput_pandas(data):
    throughput = data.groupby(['arrival_airport', 'time_bucket']).size().reset_index(name='throughput')
    return throughput

def merge_datasets_pandas(fuser, taf):
    fuser['time_bucket'] = fuser['time_bucket'].astype('datetime64[ns]')
    if taf is not None:
        taf['time_bucket'] = taf['time_bucket'].astype('datetime64[ns]')
        taf = taf.rename(columns={'station_id': 'arrival_airport'})
        data = pd.merge(fuser, taf, on=['arrival_airport', 'time_bucket'], how='left', suffixes=('', '_taf'))
    else:
        data = fuser
    data = reduce_memory_usage(data)
    return data

def feature_engineering_pandas(data):
    data['arrival_airport'] = data['arrival_airport'].astype('category')
    data['arrival_airport_encoded'] = data['arrival_airport'].cat.codes
    data['hour'] = data['time_bucket'].dt.hour
    data['day_of_week'] = data['time_bucket'].dt.dayofweek
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    data = data.sort_values(['arrival_airport', 'time_bucket'])
    data['throughput_lag_1'] = data.groupby('arrival_airport')['throughput'].shift(1)
    data['throughput_lag_1'] = data['throughput_lag_1'].bfill()
    data = reduce_memory_usage(data)
    return data

print("Processing and merging training data...")
fuser_train_files = [os.path.join(processed_data_dir, 'fuser_train', f) for f in os.listdir(os.path.join(processed_data_dir, 'fuser_train')) if f.endswith('.parquet')]
taf_train_files = [os.path.join(processed_data_dir, 'taf_train', f) for f in os.listdir(os.path.join(processed_data_dir, 'taf_train')) if f.endswith('.parquet')]
process_and_merge_data_pandas(fuser_train_files, taf_train_files, 'processed_train_data_chunk')

print("Processing and merging test data...")
fuser_test_files = [os.path.join(processed_data_dir, 'fuser_test', f) for f in os.listdir(os.path.join(processed_data_dir, 'fuser_test')) if f.endswith('.parquet')]
taf_test_files = [os.path.join(processed_data_dir, 'taf_test', f) for f in os.listdir(os.path.join(processed_data_dir, 'taf_test')) if f.endswith('.parquet')]
process_and_merge_data_pandas(fuser_test_files, taf_test_files, 'processed_test_data_chunk')

print("Loading processed training data...")
train_data_files = [f for f in os.listdir('.') if f.startswith('processed_train_data_chunk') and f.endswith('.parquet')]
if train_data_files:
    dfs = [pd.read_parquet(f) for f in train_data_files]
    train_data = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
else:
    print("No processed training data found.")
    train_data = pd.DataFrame()

print("Loading processed test data...")
test_data_files = [f for f in os.listdir('.') if f.startswith('processed_test_data_chunk') and f.endswith('.parquet')]
if test_data_files:
    dfs = [pd.read_parquet(f) for f in test_data_files]
    test_data = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
else:
    print("No processed test data found.")
    test_data = pd.DataFrame()

if not train_data.empty:
    target = 'throughput'
    if target not in train_data.columns:
        raise ValueError("Please ensure the target column is correctly specified.")
    feature_cols = [
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'throughput_lag_1', 'arrival_airport_encoded',
        'forecast_temperature', 'forecast_visibility', 'forecast_wind_speed'
    ]
    train_data[feature_cols] = train_data[feature_cols].fillna(method='ffill').fillna(method='bfill')
    test_data[feature_cols] = test_data[feature_cols].fillna(method='ffill').fillna(method='bfill')
    X = train_data[feature_cols]
    y = train_data[target]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(test_data[feature_cols])

    #Build and Train Models 

    # PyTorch Neural Network
    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y=None):
            self.X = torch.tensor(X, dtype=torch.float32)
            if y is not None:
                self.y = torch.tensor(y.values, dtype=torch.float32)
            else:
                self.y = None

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            if self.y is not None:
                return self.X[idx], self.y[idx]
            else:
                return self.X[idx]

    class FFNNModel(nn.Module):
        def __init__(self, input_size):
            super(FFNNModel, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(64, 32)
            self.relu3 = nn.ReLU()
            self.output = nn.Linear(32, 1)

        def forward(self, x):
            x = self.relu1(self.bn1(self.fc1(x)))
            x = self.relu2(self.bn2(self.fc2(x)))
            x = self.relu3(self.fc3(x))
            x = self.output(x)
            return x.squeeze()

    train_dataset = TimeSeriesDataset(X_train_scaled, y_train)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    input_size = X_train_scaled.shape[1]
    model = FFNNModel(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("Training Feed-forward Neural Network with PyTorch...")
    epochs = 30
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}")

    #XGBoost Model
    #XGBoost Model with Manual Early Stopping
    best_rmse = float('inf')  #Track the best RMSE
    best_iteration = 0  #Track the best iteration
    
    for i in range(1, 501):  #Simulate up to 500 boosting rounds
        # Initialize XGBoost model for each round
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            n_estimators=i,  # Incremental number of estimators
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=1,
            n_jobs=-1
        )
        #Train the model, I have the test set here from the processed and merged trained data to know how the model is performing
        xgb_model.fit(X_train_scaled, y_train)
    
        #Predict on validation set
        predictions = xgb_model.predict(X_valid_scaled)
        
        # Calculate RMSE for validation
        rmse = np.sqrt(mean_squared_error(y_valid, predictions))
    
        #Check for improvement
        if rmse < best_rmse:
            best_rmse = rmse  # Update best RMSE
            best_iteration = i  # Update best iteration
        else:
            print(f"Early stopping at round {i}. Best RMSE: {best_rmse:.4f} at iteration {best_iteration}.")
            break  # Stop training if no improvement
    
    print(f"Best RMSE: {best_rmse:.4f} at {best_iteration} iterations.")
    
    # Use the best model for predictions
    print("Generating predictions on validation set...")
    xgb_preds_valid = xgb_model.predict(X_valid_scaled)  # XGBoost predictions
    
    # Ensemble with Neural Network predictions
    model.eval()
    with torch.no_grad():
        X_valid_tensor = torch.tensor(X_valid_scaled, dtype=torch.float32)
        nn_preds_valid = model(X_valid_tensor).numpy()
    
    ensemble_preds_valid = (0.5 * nn_preds_valid) + (0.5 * xgb_preds_valid)
    
    # Function to calculate RMSE
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate RMSE for NN, XGBoost, and ensemble
    rmse_nn = rmse(y_valid, nn_preds_valid)
    rmse_xgb = rmse(y_valid, xgb_preds_valid)
    rmse_ensemble = rmse(y_valid, ensemble_preds_valid)
    
    # Print RMSE values
    print(f"Neural Network Validation RMSE: {rmse_nn:.4f}")
    print(f"XGBoost Validation RMSE: {rmse_xgb:.4f}")
    print(f"Ensemble Validation RMSE: {rmse_ensemble:.4f}")
    
    # Calculate final  # The score here is based on thetest data from the processed training data. The actual test TAF and Fuser data will be used for the throughput prediction(target variable) for submission
    K = 10
    score_nn = np.exp(-rmse_nn / K)
    score_xgb = np.exp(-rmse_xgb / K)
    score_ensemble = np.exp(-rmse_ensemble / K)
    
    print(f"Neural Network Final Score: {score_nn:.4f}")
    print(f"XGBoost Final Score: {score_xgb:.4f}")
    print(f"Ensemble Final Score: {score_ensemble:.4f}")
    
    #Save models and scaler
    print("Saving models and scaler...")
    torch.save(model.state_dict(), 'nn_model.pth')  # Save Neural Network model
    xgb_model.save_model('xgb_model.json')  # Save XGBoost model
    joblib.dump(scaler, 'scaler.save')  # Save scaler
    print("Processing complete!")
