import pandas as pd
import numpy as np
import os
import warnings

# Machine learning imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime

warnings.filterwarnings('ignore')

# Function to safely read CSV files with error handling
def safe_read_csv(file_path):
    try:
        # Try reading with default settings
        df = pd.read_csv(file_path)
        return df, None
    except Exception as e1:
        try:
            # Try reading with explicit encoding
            df = pd.read_csv(file_path, encoding='utf-8')
            return df, None
        except Exception as e2:
            try:
                # Try reading with different separator
                df = pd.read_csv(file_path, sep='\t')
                return df, None
            except Exception as e3:
                return None, f"Failed to read {file_path}: {str(e3)}"

# Display progress information
print("Starting advanced weather forecasting model...")

# Load datasets with error handling
print("Loading training data...")
train, train_error = safe_read_csv('train.csv')
if train_error:
    print(train_error)
    exit(1)

print("Loading test data...")
test, test_error = safe_read_csv('test.csv')
if test_error:
    print(test_error)
    exit(1)

print(f"Train data shape: {train.shape}")
print(f"Test data shape: {test.shape}")

# Validate the required columns exist
required_train_cols = ['ID', 'Year', 'Month', 'Day', 'kingdom', 'Avg_Temperature', 
                       'Radiation', 'Rain_Amount', 'Wind_Speed', 'Wind_Direction']
required_test_cols = ['ID', 'Year', 'Month', 'Day', 'kingdom']

# Check train columns
missing_train_cols = [col for col in required_train_cols if col not in train.columns]
if missing_train_cols:
    print(f"Error: Training data missing required columns: {missing_train_cols}")
    print(f"Available columns: {train.columns.tolist()}")
    exit(1)

# Check test columns
missing_test_cols = [col for col in required_test_cols if col not in test.columns]
if missing_test_cols:
    print(f"Error: Test data missing required columns: {missing_test_cols}")
    print(f"Available columns: {test.columns.tolist()}")
    exit(1)

# Clean and prepare data
print("Preparing data...")

# Handling duplicate rows if any
train = train.drop_duplicates()
test = test.drop_duplicates()

# Check for and remove rows with all missing values
train = train.dropna(how='all')
test = test.dropna(how='all')

# Ensure numeric values are properly formatted
for col in ['Year', 'Month', 'Day']:
    train[col] = pd.to_numeric(train[col], errors='coerce')
    test[col] = pd.to_numeric(test[col], errors='coerce')

# Drop rows with invalid date components
train = train.dropna(subset=['Year', 'Month', 'Day'])
test = test.dropna(subset=['Year', 'Month', 'Day'])

# Convert to appropriate types
train['Year'] = train['Year'].astype(int)
train['Month'] = train['Month'].astype(int)
train['Day'] = train['Day'].astype(int)
test['Year'] = test['Year'].astype(int)
test['Month'] = test['Month'].astype(int)
test['Day'] = test['Day'].astype(int)

# Handle the temperature unit issue - convert Kelvin to Celsius
print("Converting temperature units...")
temp_mask = train['Avg_Temperature'] > 100  # Threshold to identify Kelvin values
train.loc[temp_mask, 'Avg_Temperature'] = train.loc[temp_mask, 'Avg_Temperature'] - 273.15

# Create date keys for sorting
train['DateKey'] = train['Year']*10000 + train['Month']*100 + train['Day']
test['DateKey'] = test['Year']*10000 + test['Month']*100 + test['Day']

# Manually create derived date features instead of using pd.to_datetime
print("Creating temporal features...")
for df in [train, test]:
    # Manually calculate day of year (approximation)
    df['DayOfYear'] = (df['Month'] - 1) * 30 + df['Day']
    
    # Calculate day of week (rough approximation, sufficient for modeling)
    # This is just for feature creation, not actual calendar accuracy
    df['DayOfWeek'] = (df['DateKey'] % 7).astype(int)
    df['IsWeekend'] = df['DayOfWeek'].isin([0, 6]).astype(int)
    
    # Calculate week of year (approximation)
    df['WeekOfYear'] = (df['DayOfYear'] / 7).astype(int) + 1
    
    # Cyclic features for seasonal patterns
    df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day']/31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day']/31)
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear']/366)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear']/366)
    
    # Detailed seasons
    df['Season'] = ((df['Month'] % 12 + 3) // 3).astype(int)
    df['Season_sin'] = np.sin(2 * np.pi * df['Season']/4)
    df['Season_cos'] = np.cos(2 * np.pi * df['Season']/4)
    
    # Create month-day combination for seasonal patterns
    df['MonthDay_Combined'] = df['Month'] * 100 + df['Day']

# Encode categorical variables
print("Encoding categorical variables...")
try:
    le = LabelEncoder()
    # Handle potential encoding errors by forcing string type
    train['kingdom_str'] = train['kingdom'].astype(str)
    test['kingdom_str'] = test['kingdom'].astype(str)
    train['kingdom_encoded'] = le.fit_transform(train['kingdom_str'])
    test['kingdom_encoded'] = le.transform(test['kingdom_str'])
    
    # Create one-hot encoding for kingdoms if there aren't too many
    if len(train['kingdom'].unique()) < 20:  # Limit to prevent too many dummy variables
        kingdom_dummies = pd.get_dummies(train['kingdom'], prefix='kingdom')
        train = pd.concat([train, kingdom_dummies], axis=1)
        
        # Ensure test has same columns
        test_kingdom_dummies = pd.get_dummies(test['kingdom'], prefix='kingdom')
        for col in kingdom_dummies.columns:
            if col not in test_kingdom_dummies.columns:
                test_kingdom_dummies[col] = 0
        test = pd.concat([test, test_kingdom_dummies[kingdom_dummies.columns]], axis=1)
        
except Exception as e:
    print(f"Error encoding kingdom: {str(e)}")
    # Fallback: use numeric encoding if LabelEncoder fails
    kingdom_mapping = {k: i for i, k in enumerate(train['kingdom'].unique())}
    train['kingdom_encoded'] = train['kingdom'].map(kingdom_mapping)
    test['kingdom_encoded'] = test['kingdom'].map(kingdom_mapping)
    # Fill missing values with -1
    test['kingdom_encoded'] = test['kingdom_encoded'].fillna(-1).astype(int)

# Target columns
target_cols = ['Avg_Temperature', 'Radiation', 'Rain_Amount', 'Wind_Speed', 'Wind_Direction']

# Create a submission DataFrame
submission = pd.DataFrame({'ID': test['ID']})

# Calculate kingdom-specific statistics (more detailed)
print("Calculating advanced kingdom statistics...")
for season in range(1, 5):
    season_train = train[train['Season'] == season]
    
    # Only proceed if we have data for this season
    if len(season_train) > 0:
        kingdom_season_data = season_train.groupby('kingdom').agg({
            'Avg_Temperature': ['mean', 'median', 'std', 'min', 'max'],
            'Radiation': ['mean', 'median', 'std', 'min', 'max'],
            'Rain_Amount': ['mean', 'median', 'std', 'min', 'max'],
            'Wind_Speed': ['mean', 'median', 'std', 'min', 'max'],
            'Wind_Direction': ['mean', 'median', 'std', 'min', 'max']
        })
        
        # Flatten the multi-index columns
        kingdom_season_data.columns = [f'{col[0]}_{col[1]}_season{season}' for col in kingdom_season_data.columns.values]
        kingdom_season_data = kingdom_season_data.reset_index()
        
        # Merge to test data
        test = pd.merge(test, kingdom_season_data, on='kingdom', how='left')

# Calculate overall kingdom statistics
kingdom_data = train.groupby('kingdom').agg({
    'Avg_Temperature': ['mean', 'median', 'std', 'min', 'max'],
    'Radiation': ['mean', 'median', 'std', 'min', 'max'],
    'Rain_Amount': ['mean', 'median', 'std', 'min', 'max'],
    'Wind_Speed': ['mean', 'median', 'std', 'min', 'max'],
    'Wind_Direction': ['mean', 'median', 'std', 'min', 'max']
})

# Flatten the multi-index columns
kingdom_data.columns = ['_'.join(col).strip() for col in kingdom_data.columns.values]
kingdom_data = kingdom_data.reset_index()

# Calculate monthly statistics
monthly_data = train.groupby(['Month']).agg({
    'Avg_Temperature': ['mean', 'median', 'std'],
    'Radiation': ['mean', 'median', 'std'],
    'Rain_Amount': ['mean', 'median', 'std'], 
    'Wind_Speed': ['mean', 'median', 'std'],
    'Wind_Direction': ['mean', 'median', 'std']
}).reset_index()

# Flatten the multi-index columns
monthly_data.columns = ['Month' if col[0] == 'Month' else f'{col[0]}_monthly_{col[1]}' 
                        for col in monthly_data.columns.values]

# Calculate monthday statistics
monthday_data = train.groupby(['Month', 'Day']).agg({
    'Avg_Temperature': 'mean',
    'Radiation': 'mean',
    'Rain_Amount': 'mean', 
    'Wind_Speed': 'mean',
    'Wind_Direction': 'mean'
}).reset_index()

# Lagged features for day of year
print("Creating temporal lag features...")
lag_features = {}

# Create day-of-year lag features
for target in target_cols:
    # Create a mapping of day-of-year to average target value
    doy_map = train.groupby('DayOfYear')[target].mean().to_dict()
    
    # Add lags and future values (carefully)
    lag_days = [1, 2, 3, 5, 7, 14, 30]  # Reduced number of lags
    for lag in lag_days:
        # Safely create lag features
        try:
            col_name = f'{target}_doy_lag{lag}'
            train[col_name] = train['DayOfYear'].apply(
                lambda x: doy_map.get((x - lag) % 366, np.nan))
            test[col_name] = test['DayOfYear'].apply(
                lambda x: doy_map.get((x - lag) % 366, np.nan))
            
            # Future values
            col_name = f'{target}_doy_future{lag}'
            train[col_name] = train['DayOfYear'].apply(
                lambda x: doy_map.get((x + lag) % 366, np.nan))
            test[col_name] = test['DayOfYear'].apply(
                lambda x: doy_map.get((x + lag) % 366, np.nan))
        except Exception as e:
            print(f"Error creating lag feature {lag} for {target}: {str(e)}")
            continue

# Merge all the statistics
print("Merging advanced features...")
test = pd.merge(test, kingdom_data, on='kingdom', how='left')
test = pd.merge(test, monthly_data, on='Month', how='left')
test = pd.merge(test, monthday_data, on=['Month', 'Day'], how='left', 
                suffixes=('', '_monthday_mean'))

# Fill any missing values from the merges
for col in test.columns:
    if col in test.columns and test[col].isna().any():
        if col.endswith('_mean') or col.endswith('_median'):
            # For statistic columns, fill with the average of that statistic
            test[col] = test[col].fillna(test[col].mean() if not pd.isna(test[col].mean()) else 0)
        elif col in target_cols:
            # For direct target columns, use global mean
            test[col] = test[col].fillna(train[col].mean() if not pd.isna(train[col].mean()) else 0)

# Define base features
print("Defining feature sets...")
base_features = [
    'Year', 'Month', 'Day', 'kingdom_encoded', 
    'DayOfYear', 'Season', 'WeekOfYear', 'DayOfWeek', 'IsWeekend',
    'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 
    'DayOfYear_sin', 'DayOfYear_cos', 'Season_sin', 'Season_cos'
]

# Function to train models for each target
def train_target_models(target):
    print(f"  Processing {target}...")
    
    # Define features for this target
    features = base_features.copy()
    
    # Add kingdom statistics
    for stat in ['mean', 'median', 'std', 'min', 'max']:
        col_name = f"{target}_{stat}"
        if col_name in test.columns:
            features.append(col_name)
    
    # Add seasonal kingdom statistics
    for season in range(1, 5):
        for stat in ['mean', 'median', 'std', 'min', 'max']:
            col_name = f"{target}_{stat}_season{season}"
            if col_name in test.columns:
                features.append(col_name)
    
    # Add monthly statistics
    for stat in ['mean', 'median', 'std']:
        col_name = f"{target}_monthly_{stat}"
        if col_name in test.columns:
            features.append(col_name)
    
    # Add monthday mean
    col_name = f"{target}_monthday_mean"
    if col_name in test.columns:
        features.append(col_name)
    
    # Add lag features that exist
    lag_days = [1, 2, 3, 5, 7, 14, 30]
    for lag in lag_days:
        for prefix in ['lag', 'future']:
            col_name = f'{target}_doy_{prefix}{lag}'
            if col_name in train.columns and col_name in test.columns:
                features.append(col_name)
    
    # Add one-hot encoded kingdom columns if available
    kingdom_cols = [col for col in train.columns if col.startswith('kingdom_') and col != 'kingdom_encoded' 
                  and col in test.columns]  # Ensure columns exist in test too
    features.extend(kingdom_cols)
    
    # Final feature check - keep only features present in both datasets
    features = [f for f in features if f in train.columns and f in test.columns]
    print(f"    Using {len(features)} features")
    
    # Prepare training data
    X_train = train[features].fillna(0)
    y_train = train[target].fillna(train[target].mean())
    
    # Handle potential issues with the target variable
    if y_train.isna().any():
        print(f"  Warning: Found {y_train.isna().sum()} NaN values in {target}. Filling with mean.")
        y_train = y_train.fillna(y_train.mean())
    
    # Train multiple models for ensemble
    models = []
    predictions = {}
    
    # Model 1: XGBoost
    try:
        print("    Training XGBoost model...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        models.append(('xgb', xgb_model))
        
        # Make prediction
        X_test = test[features].fillna(0)
        predictions['xgb'] = xgb_model.predict(X_test)
        print("    XGBoost model complete")
    except Exception as e:
        print(f"    Error training XGBoost for {target}: {str(e)}")
    
    # Model 2: LightGBM
    try:
        print("    Training LightGBM model...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1
        )
        lgb_model.fit(X_train, y_train)
        models.append(('lgb', lgb_model))
        
        # Make prediction
        X_test = test[features].fillna(0)
        predictions['lgb'] = lgb_model.predict(X_test)
        print("    LightGBM model complete")
    except Exception as e:
        print(f"    Error training LightGBM for {target}: {str(e)}")
    
    # Model 3: Random Forest (as a reliable fallback)
    try:
        print("    Training Random Forest model...")
        rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models.append(('rf', rf_model))
        
        # Make prediction
        X_test = test[features].fillna(0)
        predictions['rf'] = rf_model.predict(X_test)
        print("    Random Forest model complete")
    except Exception as e:
        print(f"    Error training RandomForest for {target}: {str(e)}")
    
    # Calculate ensemble prediction (average of all models)
    if predictions:
        print(f"    Creating ensemble prediction from {len(predictions)} models")
        ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
        return ensemble_pred
    else:
        # Fallback to kingdom means
        print(f"    Warning: No successful models for {target}. Using fallback.")
        return test[f"{target}_mean"].values

print("Creating advanced model ensemble...")
# Process each target column
for target in target_cols:
    predictions = train_target_models(target)
    submission[target] = predictions

# Post-process predictions to ensure physical constraints
print("Post-processing predictions...")
submission['Radiation'] = submission['Radiation'].clip(lower=0)
submission['Rain_Amount'] = submission['Rain_Amount'].clip(lower=0)
submission['Wind_Speed'] = submission['Wind_Speed'].clip(lower=0)
submission['Wind_Direction'] = submission['Wind_Direction'] % 360

# Save submission file
print("Saving submission file...")
submission.to_csv('submission.csv', index=False)
print("Enhanced submission file created successfully!")