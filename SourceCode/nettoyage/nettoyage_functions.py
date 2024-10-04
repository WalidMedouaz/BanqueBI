import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def extract_year(date_str):
    """
    Fonction pour extraire l'année d'une chaîne de date en format YYYY-MM-DD ou DD/MM/YYYY.
    Retourne NaN si la date est invalide.
    """
    try:
        # Gérer le format YYYY-MM-DD
        if '-' in date_str:
            return int(date_str.split('-')[0])
        # Gérer le format DD/MM/YYYY
        elif '/' in date_str:
            return int(date_str.split('/')[-1])
        else:
            return np.nan
    except:
        return np.nan


def nettoyage(datanames):
    # Lire les données de table1.csv
    data = pd.read_csv(datanames[0])

    # Lire les données de table2.csv
    data2 = pd.read_csv(datanames[1])
    
    # Supprimer les colonnes dans data si elles ne sont pas pertinentes pour l'analyse
    data.drop(columns=['ID'], inplace=True)
    data.drop(columns=['CDSEXE'], inplace=True)
    data.drop(columns=['DTADH'], inplace=True)
    data.drop(columns=['CDDEM'], inplace=True)
    data.drop(columns=['DTDEM'], inplace=True)
    data.drop(columns=['CDMOTDEM'], inplace=True)
    data.drop(columns=['AGEAD'], inplace=True)
    data.drop(columns=['RANGAGEAD'], inplace=True)
    data.drop(columns=['RANGDEM'], inplace=True)
    data.drop(columns=['RANGADH'], inplace=True)
    data.drop(columns=['ANNEEDEM'], inplace=True)

    # Modifier la colonne RANGAGEDEM pour qu'elle ne contiennent que l'intervalle d'âge et non plus de cette forme X  XX-XX
    data['RANGAGEDEM'] = data['RANGAGEDEM'].str.split('  ').str[1]


    # Supprimer les colonnes dans data si elles ne sont pas pertinentes pour l'analyse
    data2.drop(columns=['ID'], inplace=True)
    data2.drop(columns=['CDSEXE'], inplace=True)
    data2.drop(columns=['DTADH'], inplace=True)
    data2.drop(columns=['CDMOTDEM'], inplace=True)
    data2.drop(columns=['BPADH'], inplace=True)
    data2.drop(columns=['DTDEM'], inplace=True)

    # Ajouter la colonne AGE à data2. Si DTNAIS est à 0000-00-00, on mettra NaN, sinon on calcule l'âge en 2007
    data2['AGE'] = data2['DTNAIS'].apply(lambda x: 2007 - extract_year(x) if x != '0000-00-00' else np.nan)

    # Ajouter la colonne RANGAGEDEM à data2
    data2['RANGAGEDEM'] = data2['AGE'].apply(age_interval)

    # Retourner la data nettoyée
    return data, data2

# permet de calculer l'intervalle RANGEAGEDEM en fonction de l'âge pour data2
def age_interval(age):
    if 19 <= age <= 25:
        return "19-25"
    elif 26 <= age <= 30:
        return "26-30"
    elif 31 <= age <= 35:
        return "31-35"
    elif 36 <= age <= 40:
        return "36-40"
    elif 41 <= age <= 45:
        return "41-45"
    elif 46 <= age <= 50:
        return "46-50"
    elif 51 <= age <= 55:
        return "51-55"
    elif 56 <= age <= 60:
        return "56-60"
    elif 61 <= age <= 65:
        return "61-65"
    elif 66 <= age <= 70:
        return "66-70"
    elif age >= 71:
        return "71+"
    else:
        return np.nan  # Si l'âge est en dehors des intervalles gérés



