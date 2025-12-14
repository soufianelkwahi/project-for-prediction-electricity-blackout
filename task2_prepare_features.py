# Task 2: Prepare features, one-hot encoding, and create sequences for LSTM
import numpy as np

# Load real data
df = pd.read_csv("solar_data_pvgis/all_cities_real_data.csv")
df['date'] = pd.to_datetime(df['date'])

# One-hot encoding for cities
df = pd.get_dummies(df, columns=['city'], dtype=float)

# Define features (real radiation and temperature + rolling + one-hot)
features = ['mean_rad', 'temp_max', 'rad_rolling', 'temp_rolling'] + \
           [col for col in df.columns if col.startswith('city_')]

# Sequence length (7 days)
seq_len = 7
X, y, indices = [], [], []

# Group by city (using one-hot columns)
city_cols = [col for col in df.columns if col.startswith('city_')]

for _, group in df.groupby(city_cols, sort=False):
    group_features = group[features].values
    
    # Create target label (outage probability based on real data)
    # Example logic: low radiation = higher outage chance
    prob = 0.05 + 0.4 * (group['mean_rad'] < 3.5) + 0.3 * (group['temp_max'] > 30)
    prob = np.clip(prob, 0, 1)
    labels = np.random.binomial(1, prob.values)  # Simulate binary outage
    
    # Create sequences
    for i in range(seq_len, len(group)):
        X.append(group_features[i-seq_len:i])
        y.append(labels[i])
        indices.append(group.index[i])

X = np.array(X)
y = np.array(y)

print(f"Sequences created: {X.shape} (samples, time_steps, features)")
print(f"Target shape: {y.shape}")