import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os
import warnings
import lightgbm as lgb
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
print("Starting weather forecasting model...")

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

# Create date keys for sorting (avoid datetime conversion issues)
train['DateKey'] = train['Year']*10000 + train['Month']*100 + train['Day']
test['DateKey'] = test['Year']*10000 + test['Month']*100 + test['Day']

# Generate temporal features
print("Creating features...")
for df in [train, test]:
    # Create approximation for day of year
    df['DayOfYear'] = (df['Month'] - 1) * 30 + df['Day']
    df['MonthDay'] = df['Day']
    df['Season'] = (df['Month'] % 12 + 3) // 3  # 1: Spring, 2: Summer, 3: Fall, 4: Winter
    # Create month-day combination for seasonal patterns
    df['MonthDay_Combined'] = df['Month'] * 100 + df['Day']

# Encode categorical variables
try:
    le = LabelEncoder()
    # Handle potential encoding errors by forcing string type
    train['kingdom_str'] = train['kingdom'].astype(str)
    test['kingdom_str'] = test['kingdom'].astype(str)
    train['kingdom_encoded'] = le.fit_transform(train['kingdom_str'])
    test['kingdom_encoded'] = le.transform(test['kingdom_str'])
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

# Merge kingdom statistics to test data
print("Merging features...")
test = pd.merge(test, kingdom_data, on='kingdom', how='left')

# Merge month-day statistics to test data
test = pd.merge(test, monthday_data, on=['Month', 'Day'], how='left', 
                suffixes=('', '_monthday_mean'))

# Fill any missing values from the merges
for col in test.columns:
    if test[col].isna().any():
        if col.endswith('_mean') or col.endswith('_median'):
            # For statistic columns, fill with the average of that statistic
            test[col] = test[col].fillna(test[col].mean())
        elif col in target_cols:
            # For direct target columns, use global mean
            test[col] = test[col].fillna(train[col].mean())

# Define features based on what's available
print("Defining feature sets...")
base_features = ['Year', 'Month', 'Day', 'kingdom_encoded', 'DayOfYear', 'Season']

# For each target, train a model and make predictions
print("Training models and making predictions...")
for target in target_cols:
    print(f"  Processing {target}...")
    
    # Define features for this target
    features = base_features.copy()
    
    # Add kingdom statistics
    for stat in ['mean', 'median', 'std']:
        col_name = f"{target}_{stat}"
        if col_name in test.columns:
            features.append(col_name)
    
    # Add monthday mean
    col_name = f"{target}_monthday_mean"
    if col_name in test.columns:
        features.append(col_name)
    
    # Keep only features present in both datasets
    features = [f for f in features if f in train.columns and f in test.columns]
    
    # Prepare training data
    X_train = train[features].fillna(0)
    y_train = train[target].fillna(train[target].mean())
    
    # Handle potential issues with the target variable
    if y_train.isna().any():
        print(f"  Warning: Found {y_train.isna().sum()} NaN values in {target}. Filling with mean.")
        y_train = y_train.fillna(y_train.mean())
    
    # Train model with robust parameters
    model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1
        )
    
    try:
        model.fit(X_train, y_train)
        
        # Prepare test data
        X_test = test[features].fillna(0)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Add to submission DataFrame
        submission[target] = predictions
    except Exception as e:
        print(f"  Error training model for {target}: {str(e)}")
        # Use fallback: kingdom mean
        submission[target] = test[f"{target}_mean"]

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