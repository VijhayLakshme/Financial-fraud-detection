import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def load_and_preprocess_data(file):
    import streamlit as st

    # ✅ Read only first 10,000 rows
    df = pd.read_csv(file, nrows=10000)

    # ✅ Fill missing values
    df.fillna(0, inplace=True)

    # ✅ Add synthetic fraud labels if missing
    if 'is_fraud' not in df.columns:
        st.warning("⚠ 'is_fraud' column not found — adding synthetic labels for testing.")
        df['is_fraud'] = np.random.choice([0, 1], size=len(df), p=[0.97, 0.03])

    # ✅ Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # ✅ Split features and target
    features = df.drop('is_fraud', axis=1)
    target = df['is_fraud']

    # ✅ Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, target, df

def train_models(X_train, y_train):
    models = {}
    models['LogisticRegression'] = LogisticRegression().fit(X_train, y_train)
    models['RandomForest'] = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)
    models['IsolationForest'] = IsolationForest(contamination=0.01).fit(X_train)
    return models

def train_autoencoder(X_train):
    input_layer = Input(shape=(X_train.shape[1],))
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(encoded)
    output = Dense(X_train.shape[1], activation='linear')(decoded)

    autoencoder = Model(input_layer, output)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, verbose=0)

    return autoencoder
