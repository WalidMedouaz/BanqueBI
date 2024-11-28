
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



def convert(data):

    df = data.copy()

    # Convertir 'DTDEM' en variable cible 'dem_bool'
    df['dem_bool'] = df['DTDEM'].apply(lambda x: 1 if pd.notnull(x) else 0)

    # Remplacer les valeurs NaN dans AGE avec une estimation ou une valeur moyenne
    df['AGE'] = df['AGE'].fillna(df['AGE'].mean())

    # Supprimer ou encoder les colonnes inutiles comme 'DTDEM' si elle est redondante
    # df = df.drop(columns=['DTDEM', 'DTADH', 'CDMOTDEM', 'DTNAIS'])

    # Recoder/supprimer les colonnes non numériques soit : DTADH, DTDEM, CDMOTDEM, DTNAIS
    df = df.drop(columns=['DTADH', 'DTDEM', 'CDMOTDEM', 'DTNAIS'])

    # Encodage des variables catégorielles
    df = pd.get_dummies(df, columns=['CDSEXE', 'CDSITFAM', 'CDTMT', 'CDCATCL'])

    # On réduit la taille des données réduisant le nombre de démissionnaires pour équilibrer les classes
    # On prend tous les non-démissionnaires et on prend un échantillon aléatoire des démissionnaires de même taille
    dem = df[df['dem_bool'] == 0]
    print(dem.info())
    no_dem = df[df['dem_bool'] == 1]
    print(no_dem.info())
    df2 = pd.concat([dem, no_dem])

    # s'assurer qu'aucune valeur n'est à Nan
    df2 = df.fillna(0)


    return df2
