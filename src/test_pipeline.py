import pandas as pd
from pipeline import split_dataset 

def test_split_dataset_with_csv():
    # Chargement du fichier CSV
    df = pd.read_csv('data3.csv')

    # Split
    X_train, X_test, y_train, y_test = split_dataset(df, target_col='Churn')

    # Vérifications
    assert len(X_train) == len(y_train), "X_train et y_train doivent avoir la même longueur"
    assert len(X_test) == len(y_test), "X_test et y_test doivent avoir la même longueur"
    assert X_train.shape[1] == df.shape[1] - 1, "X doit contenir toutes les colonnes sauf la cible"