import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
# pipeline.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

#  1. Chargement et nettoyage
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df



#  2. Encodage des colonnes catégorielles
def encode_data(df, target_col='Churn'):
    df = df.copy()
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col != target_col:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df[target_col] = LabelEncoder().fit_transform(df[target_col].astype(str))
    return df


#  3. Normalisation des colonnes numériques
def normalize_features(df, numeric_cols):
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler

#  4. Split Train/Test
def split_dataset(df, target_col='Churn'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)


#5. Entraînement du modèle
def train_model(model_name, X_train, y_train):
    if model_name == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_name == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Modèle non reconnu : 'logistic' ou 'rf'")
    model.fit(X_train, y_train)
    return model

# 6. Évaluation du modèle
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
