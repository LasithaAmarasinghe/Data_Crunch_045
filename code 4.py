import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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
print("Starting weather forecasting model with LSTM...")

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

# Generate temporal features
print("Creating features...")
for df in [train, test]:
    # Create approximation for day of year
    df['DayOfYear'] = (df['Month'] - 1) * 30 + df['Day']
    df['MonthDay'] = df['Day']
    df['Season'] = (df['Month'] % 12 + 3) // 3  # 1: Spring, 2: Summer, 3: Fall, 4: Winter
    # Create month-day combination for seasonal patterns
    df['MonthDay_Combined'] = df['Month'] * 100 + df['Day']
    # Create sin and cos features for cyclical variables
    df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day']/31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day']/31)
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear']/365)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear']/365)

# Encode categorical variables
try:
    le = LabelEncoder()
    # Handle potential encoding errors by forcing string type
    train['kingdom_str'] = train['kingdom'].astype(str)
    test['kingdom_str'] = test['kingdom'].astype(str)
    
    # Fit on combined data to ensure all categories are represented
    all_kingdoms = pd.concat([train['kingdom_str'], test['kingdom_str']]).unique()
    le.fit(all_kingdoms)
    
    train['kingdom_encoded'] = le.transform(train['kingdom_str'])
    test['kingdom_encoded'] = le.transform(test['kingdom_str'])
except Exception as e:
    print(f"Error encoding kingdom: {str(e)}")
    # Fallback: use numeric encoding if LabelEncoder fails
    all_kingdoms = set(train['kingdom'].unique()) | set(test['kingdom'].unique())
    kingdom_mapping = {k: i for i, k in enumerate(all_kingdoms)}
    train['kingdom_encoded'] = train['kingdom'].map(kingdom_mapping)
    test['kingdom_encoded'] = test['kingdom'].map(kingdom_mapping)
    # Fill missing values with -1
    test['kingdom_encoded'] = test['kingdom_encoded'].fillna(-1).astype(int)

# Target columns
target_cols = ['Avg_Temperature', 'Radiation', 'Rain_Amount', 'Wind_Speed', 'Wind_Direction']

# Create a submission DataFrame
submission = pd.DataFrame({'ID': test['ID']})

# Prepare data for LSTM
print("Preparing data for LSTM...")

# Define features for LSTM
lstm_features = ['Year', 'Month', 'Day', 'kingdom_encoded', 'DayOfYear', 'Season',
                'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 'DayOfYear_sin', 'DayOfYear_cos']

# Sort data chronologically and by kingdom for sequence creation
train = train.sort_values(['kingdom_encoded', 'DateKey'])
test = test.sort_values(['kingdom_encoded', 'DateKey'])

# Create sequences for each kingdom - FIXED VERSION
def create_sequences(data, feature_cols, target_cols, seq_length=7):
    """Create input sequences for LSTM model."""
    X, y = [], []
    kingdoms = data['kingdom_encoded'].unique()
    
    for kingdom in kingdoms:
        kingdom_data = data[data['kingdom_encoded'] == kingdom]
        if len(kingdom_data) <= seq_length:
            continue
            
        features = kingdom_data[feature_cols].values
        
        # Extract target values from the data
        # This gets the actual target columns instead of using indices
        targets = kingdom_data[target_cols].values
        
        for i in range(len(kingdom_data) - seq_length):
            X.append(features[i:i+seq_length])
            y.append(targets[i+seq_length])
    
    return np.array(X), np.array(y)

# Fill missing values in targets with the mean
for col in target_cols:
    if train[col].isna().any():
        mean_value = train[col].mean()
        train[col] = train[col].fillna(mean_value)

# Fill missing values in features
for col in lstm_features:
    if col in train.columns and train[col].isna().any():
        train[col] = train[col].fillna(0)
    if col in test.columns and test[col].isna().any():
        test[col] = test[col].fillna(0)

# Scale features for better LSTM performance
print("Scaling features...")
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Create dataframes with features and targets for scaling
train_features_df = train[lstm_features]
train_targets_df = train[target_cols]

# Fit scalers on training data
train_features_scaled = feature_scaler.fit_transform(train_features_df)
train_targets_scaled = target_scaler.fit_transform(train_targets_df)

# Create scaled dataframes
train_features_scaled_df = pd.DataFrame(train_features_scaled, columns=lstm_features)
train_targets_scaled_df = pd.DataFrame(train_targets_scaled, columns=target_cols)

# Add kingdom_encoded back for sequence creation
train_features_scaled_df['kingdom_encoded'] = train['kingdom_encoded'].values

# Create sequences for training
seq_length = 7  # Define sequence length (e.g., use past 7 days to predict next day)
print(f"Creating sequences with length {seq_length}...")

# Prepare training sequences - FIXED VERSION
X_train, y_train = create_sequences(
    pd.concat([train_features_scaled_df, train_targets_scaled_df], axis=1),
    lstm_features,
    target_cols,  # Use actual column names
    seq_length
)

print(f"Training sequences shape: {X_train.shape}, {y_train.shape}")

# Build LSTM model
print("Building LSTM model...")
def build_lstm_model(seq_length, n_features, n_outputs):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(n_outputs)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train model
lstm_model = build_lstm_model(seq_length, len(lstm_features), len(target_cols))
print(lstm_model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Training LSTM model...")
history = lstm_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Prepare test data for prediction
print("Preparing test data for prediction...")

# Scale test features
test_features_scaled = feature_scaler.transform(test[lstm_features])
test_features_scaled_df = pd.DataFrame(test_features_scaled, columns=lstm_features)
test_features_scaled_df['kingdom_encoded'] = test['kingdom_encoded'].values

# Function to get latest sequence for each kingdom - FIXED VERSION
def get_kingdom_sequences(train_features_df, train_targets_df, test_features_df, kingdom_list, feature_cols, seq_length):
    all_sequences = {}
    
    # Combine features and targets for train data
    train_combined = pd.concat([train_features_df, train_targets_df], axis=1)
    
    for kingdom in kingdom_list:
        # Get train data for this kingdom
        kingdom_train = train_combined[train_combined['kingdom_encoded'] == kingdom]
        kingdom_test = test_features_df[test_features_df['kingdom_encoded'] == kingdom]
        
        if len(kingdom_train) < seq_length and len(kingdom_train) > 0:
            # Not enough history, use what we have and pad with repeats
            sequence = None
        elif len(kingdom_train) >= seq_length:
            # Enough history to create a sequence
            sequence = kingdom_train[feature_cols].values[-seq_length:]
        else:
            # No history for this kingdom
            sequence = None
        
        all_sequences[kingdom] = {
            'sequence': sequence,
            'test_indices': kingdom_test.index.tolist() if not kingdom_test.empty else []
        }
    
    return all_sequences

# Get sequences for each kingdom in test data
kingdom_sequences = get_kingdom_sequences(
    train_features_scaled_df,
    train_targets_scaled_df,
    test_features_scaled_df,
    test['kingdom_encoded'].unique(),
    lstm_features,
    seq_length
)

# Function to make predictions for test data - FIXED VERSION
def predict_test_data(model, kingdom_seqs, test_df):
    predictions = np.zeros((len(test_df), len(target_cols)))
    kingdoms_without_sequences = []
    
    # For each kingdom, use its sequence to predict
    for kingdom, data in kingdom_seqs.items():
        if data['sequence'] is None or len(data['test_indices']) == 0:
            kingdoms_without_sequences.append(kingdom)
            continue
            
        # Get indices for this kingdom in test data
        indices = data['test_indices']
        
        # Use the sequence to predict for all days of this kingdom
        sequence = data['sequence']
        kingdom_preds = model.predict(np.array([sequence]), verbose=0)
        
        # Assign predictions to all test days for this kingdom
        for idx in indices:
            predictions[idx] = kingdom_preds[0]
    
    if kingdoms_without_sequences:
        print(f"Warning: No sequences available for kingdoms: {kingdoms_without_sequences}")
    
    return predictions

# Make predictions
print("Making predictions...")
# First attempt with kingdom sequences
pred_scaled = predict_test_data(lstm_model, kingdom_sequences, test_features_scaled_df)

# For any missing predictions (no sequence available), use fallback
missing_mask = np.all(pred_scaled == 0, axis=1)
if np.any(missing_mask):
    print(f"Warning: {np.sum(missing_mask)} test entries have no predictions. Using fallback.")
    
    # Calculate kingdom means for fallback
    kingdom_means = train.groupby('kingdom_encoded')[target_cols].mean().reset_index()
    kingdom_means_dict = {row['kingdom_encoded']: row[target_cols].values 
                         for _, row in kingdom_means.iterrows()}
    
    # Fill missing predictions with kingdom means
    for i in range(len(pred_scaled)):
        if missing_mask[i]:
            kingdom = test.iloc[i]['kingdom_encoded']
            if kingdom in kingdom_means_dict:
                # Scale the mean values
                mean_values = kingdom_means_dict[kingdom].reshape(1, -1)
                pred_scaled[i] = target_scaler.transform(mean_values)[0]
            else:
                # Use global means as last resort
                global_means = train[target_cols].mean().values.reshape(1, -1)
                pred_scaled[i] = target_scaler.transform(global_means)[0]

# Inverse transform predictions to original scale
predictions = target_scaler.inverse_transform(pred_scaled)

# Add predictions to submission DataFrame
for i, col in enumerate(target_cols):
    submission[col] = predictions[:, i]

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