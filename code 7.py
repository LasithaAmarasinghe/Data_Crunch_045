import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings
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
print("Starting stacked weather forecasting model...")

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

# Data preprocessing (same as before)
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

# Generate temporal features
print("Creating features...")
for df in [train, test]:
    # Create date key for sorting
    df['DateKey'] = df['Year']*10000 + df['Month']*100 + df['Day']
    
    # Create approximation for day of year
    df['DayOfYear'] = (df['Month'] - 1) * 30 + df['Day']
    
    # Season encoding (1: Spring, 2: Summer, 3: Fall, 4: Winter)
    df['Season'] = (df['Month'] % 12 + 3) // 3
    
    # Cyclical encoding for month and day
    df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day']/31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day']/31)
    
    # Day of week approximation (not perfect but can help)
    df['DayOfWeek'] = df['DateKey'] % 7
    
    # Create month-day combination for seasonal patterns
    df['MonthDay_Combined'] = df['Month'] * 100 + df['Day']

# Encode categorical variables
print("Encoding categorical variables...")
le = LabelEncoder()
# Handle potential encoding errors by forcing string type
train['kingdom_str'] = train['kingdom'].astype(str)
test['kingdom_str'] = test['kingdom'].astype(str)
train['kingdom_encoded'] = le.fit_transform(train['kingdom_str'])
test['kingdom_encoded'] = le.transform(test['kingdom_str'])

# Target columns
target_cols = ['Avg_Temperature', 'Radiation', 'Rain_Amount', 'Wind_Speed', 'Wind_Direction']

# Create a submission DataFrame
submission = pd.DataFrame({'ID': test['ID']})

# Calculate kingdom-specific statistics
print("Calculating kingdom statistics...")
kingdom_data = train.groupby('kingdom').agg({
    'Avg_Temperature': ['mean', 'median', 'std'],
    'Radiation': ['mean', 'median', 'std'],
    'Rain_Amount': ['mean', 'median', 'std'],
    'Wind_Speed': ['mean', 'median', 'std'],
    'Wind_Direction': ['mean', 'median', 'std']
})

# Flatten the multi-index columns
kingdom_data.columns = ['_'.join(col).strip() for col in kingdom_data.columns.values]
kingdom_data = kingdom_data.reset_index()

# Calculate month-day specific statistics (seasonal patterns)
monthday_data = train.groupby(['Month', 'Day']).agg({
    'Avg_Temperature': 'mean',
    'Radiation': 'mean',
    'Rain_Amount': 'mean', 
    'Wind_Speed': 'mean',
    'Wind_Direction': 'mean'
}).reset_index()

# Calculate season statistics
season_data = train.groupby('Season').agg({
    'Avg_Temperature': 'mean',
    'Radiation': 'mean',
    'Rain_Amount': 'mean', 
    'Wind_Speed': 'mean',
    'Wind_Direction': 'mean'
}).reset_index()

# Merge kingdom statistics to train and test data
train = pd.merge(train, kingdom_data, on='kingdom', how='left')
test = pd.merge(test, kingdom_data, on='kingdom', how='left')

# Merge month-day statistics to train and test data
train = pd.merge(train, monthday_data, on=['Month', 'Day'], how='left', 
              suffixes=('', '_monthday_mean'))
test = pd.merge(test, monthday_data, on=['Month', 'Day'], how='left', 
             suffixes=('', '_monthday_mean'))

# Merge season statistics to train and test data
train = pd.merge(train, season_data, on='Season', how='left',
              suffixes=('', '_season_mean'))
test = pd.merge(test, season_data, on='Season', how='left',
             suffixes=('', '_season_mean'))

# Fill any missing values from the merges
for col in test.columns:
    if test[col].isna().any():
        if col.endswith('_mean') or col.endswith('_median') or col.endswith('_std'):
            # For statistic columns, fill with the average of that statistic
            test[col] = test[col].fillna(test[col].mean())

# Define base features based on what's available
print("Defining feature sets...")
base_features = [
    'Year', 'Month', 'Day', 'kingdom_encoded', 'DayOfYear', 'Season',
    'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 'DayOfWeek'
]

# Function to make stacked predictions
def stacked_predictions(X_train, y_train, X_test, feature_columns, target):
    """
    Use stacking to combine predictions from multiple models
    """
    print(f"  Creating stacked model for {target}...")
    
    # Define base models
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)),
        ('ridge', Ridge(alpha=1.0)),
        ('lasso', Lasso(alpha=0.1))
    ]
    
    # For storing meta features
    meta_X_train = np.zeros((X_train.shape[0], len(base_models)))
    meta_X_test = np.zeros((X_test.shape[0], len(base_models)))
    
    # k-fold for creating meta features
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Process each base model
    for i, (name, model) in enumerate(base_models):
        print(f"    Training {name} model...")
        # For test predictions, train on all data
        model.fit(X_train, y_train)
        # Make predictions on test data
        meta_X_test[:, i] = model.predict(X_test)
        
        # Create out-of-fold predictions for meta-training data
        meta_preds = np.zeros(X_train.shape[0])
        for train_idx, val_idx in kf.split(X_train):
            # Train on training folds
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            # Predict on validation fold
            meta_preds[val_idx] = model.predict(X_train.iloc[val_idx])
        
        # Store out-of-fold predictions as meta features
        meta_X_train[:, i] = meta_preds
    
    # Train meta model
    print("    Training meta model...")
    meta_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, random_state=42)
    meta_model.fit(meta_X_train, y_train)
    
    # Final prediction using meta model
    final_predictions = meta_model.predict(meta_X_test)
    
    return final_predictions

# Loop through each target column and create stacked models
print("Training stacked models and making predictions...")
for target in target_cols:
    print(f"Processing {target}...")
    
    # Define features for this target
    features = base_features.copy()
    
    # Add kingdom statistics
    for stat in ['mean', 'median', 'std']:
        col_name = f"{target}_{stat}"
        if col_name in test.columns:
            features.append(col_name)
    
    # Add monthday and season means
    col_names = [f"{target}_monthday_mean", f"{target}_season_mean"]
    features.extend([col for col in col_names if col in test.columns])
    
    # Keep only features present in both datasets
    features = [f for f in features if f in train.columns and f in test.columns]
    
    # Prepare training data
    X_train = train[features].fillna(0)
    y_train = train[target].fillna(train[target].mean())
    
    # Prepare test data
    X_test = test[features].fillna(0)
    
    try:
        # Generate stacked predictions
        predictions = stacked_predictions(X_train, y_train, X_test, features, target)
        
        # Add to submission DataFrame
        submission[target] = predictions
    except Exception as e:
        print(f"  Error making stacked predictions for {target}: {str(e)}")
        # Fallback to using kingdom mean
        if f"{target}_mean" in test.columns:
            submission[target] = test[f"{target}_mean"]
        else:
            submission[target] = y_train.mean()

# Post-process predictions to ensure physical constraints
print("Post-processing predictions...")
submission['Radiation'] = submission['Radiation'].clip(lower=0)
submission['Rain_Amount'] = submission['Rain_Amount'].clip(lower=0)
submission['Wind_Speed'] = submission['Wind_Speed'].clip(lower=0)
submission['Wind_Direction'] = submission['Wind_Direction'] % 360

# Save submission file
print("Saving submission file...")
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")