import os
# Force scikit-learn's underlying openBLAS backend to bypass thread throttling on macOS
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

print("🔄 Loading datasets...")
train_df = pd.read_csv('blackfriday_train.csv')
test_df = pd.read_csv('blackfriday_test.csv')

# --- 1. DATA CLEANING & FEATURE ENGINEERING ---
print("🛠️ Engineering features...")
def clean_and_engineer(df):
    # Handle missing sub-categories by marking them as a separate 0 class
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0).astype(int)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0).astype(int)
    
    # Map 'Stay_In_Current_City_Years' 4+ to numerical 4
    df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].str.replace('+', '', regex=False).astype(int)
    return df

train_df = clean_and_engineer(train_df)
test_df = clean_and_engineer(test_df)

# Label Encoding categorical demographics
le_dict = {}
categorical_cols = ['Gender', 'Age', 'City_Category']

for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    # Handle unseen labels in test just in case
    test_df[col] = test_df[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
    le.classes_ = np.append(le.classes_, '<unknown>')
    test_df[col] = le.transform(test_df[col])
    le_dict[col] = le

# --- 2. CUSTOMER SEGMENTATION CLUSTERING (K-MEANS) ---
print("🤖 Generating demographic clusters...")
cluster_features = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']

scaler = StandardScaler()
scaled_train_meta = scaler.fit_transform(train_df[cluster_features])
scaled_test_meta = scaler.transform(test_df[cluster_features])

# Using 3 core spend behavior personas/clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
train_df['User_Cluster'] = kmeans.fit_predict(scaled_train_meta)
test_df['User_Cluster'] = kmeans.predict(scaled_test_meta)

# --- 3. MODEL TRAINING ---
features = cluster_features + ['Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'User_Cluster']
X = train_df[features]
y = train_df['Purchase']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("🌲 Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluation
val_preds = model.predict(X_val)
rmse = root_mean_squared_error(y_val, val_preds)
r2 = r2_score(y_val, val_preds)

print(f"✅ Validation Evaluation -> RMSE: {rmse:.2f} | R2 Score: {r2:.4f}")

# Save artifacts for deployment
joblib.dump(model, 'rf_model.pkl')
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_dict, 'label_encoders.pkl')
joblib.dump(features, 'feature_names.pkl')
print("💾 All model artifacts securely saved!")