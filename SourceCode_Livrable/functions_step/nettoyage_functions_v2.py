import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Fonction pour extraire l'année d'une chaîne de date en format YYYY-MM-DD ou DD/MM/YYYY
def extract_year(date_str):
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

# Fonction pour calculer l'intervalle d'âge
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

# Fonction de nettoyage des deux tables
def nettoyage(datanames):
    # Lire les données de table1.csv et table2.csv
    data1 = pd.read_csv(datanames[0])
    data2 = pd.read_csv(datanames[1])

    # Liste des colonnes à supprimer
    cols_to_drop = ['ID', 'CDDEM', 'ANNEEDEM', 'RANGAGEAD', 'AGEDEM', 'RANGAGEDEM', 'RANGDEM', 'ADH', 'RANGADH', 'BPADH']

    # Supprimer les colonnes inutiles de table1
    data1.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Supprimer les colonnes inutiles de table2
    data2.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Traitement spécifique pour table2
    # Modifier la colonne CDMOTDEM : remplacer les valeurs manquantes par 'ND' pour les non-démissionnaires
    data2['CDMOTDEM'] = data2['CDMOTDEM'].fillna('ND')

    # Ajouter la colonne AGE dans table2 : calculer l'âge en 2007 à partir de DTNAIS
    data2['AGE'] = data2['DTNAIS'].apply(lambda x: 2007 - extract_year(x) if x != '0000-00-00' else np.nan)

    # Appel de la fonction remove_aberrant pour supprimer les valeurs aberrantes
    #data1 = remove_aberrant_data1(data1)
    data2 = remove_aberrant_data2(data2)

    # Retourner les tables nettoyées
    return data1, data2


# Fonction pour supprimer les différents types de valeurs aberrantes (appelée dans la fonction nettoyage)
def remove_aberrant_data2(data):
    
    # Si la date dans DTNAIS est 0000-00-00, supprimer la ligne
    data = data[data['DTNAIS'] != '0000-00-00']

    #print("Taille avant : ", data.shape[0])

    data = data[~((data['DTDEM'] == '31/12/1900') & (data['CDMOTDEM'] != 'ND'))]

    #print("Taille après : ", data.shape[0])

    # Si DTDEM égale à 31/12/1900 mettre la valeur à NaN
    data.loc[data['DTDEM'] == '31/12/1900', 'DTDEM'] = np.nan

    return data


