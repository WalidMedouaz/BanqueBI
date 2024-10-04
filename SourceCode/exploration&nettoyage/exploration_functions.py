import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Définition d'une classe pour structurer les résultats d'exploration
class DataFrameExploration:
    def __init__(self, column_types, unique_values, numeric_stats, missing_values, categorical_distributions):
        self.column_types = column_types
        self.unique_values = unique_values
        self.numeric_stats = numeric_stats
        self.missing_values = missing_values
        self.categorical_distributions = categorical_distributions

# Fonction pour explorer les colonnes d'un dataframe
def explore_dataframe(df):
    # Types des colonnes
    column_types = df.dtypes

    # Valeurs uniques ou plage pour les colonnes
    unique_values = {}
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            unique_values[column] = f"Plage de valeurs ({df[column].min()} - {df[column].max()})"
        else:
            values = df[column].unique()
            if len(values) <= 5:
                unique_values[column] = values
            else:
                unique_values[column] = f"Trop de valeurs uniques ({len(values)} valeurs)"

    # Statistiques sur les colonnes numériques
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if not numeric_columns.empty:
        numeric_stats = df[numeric_columns].describe()
    else:
        numeric_stats = "Pas de colonnes numériques"

    # Comptage des valeurs manquantes
    missing_values = df.isnull().sum()

    # Distribution des colonnes catégorielles
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    categorical_distributions = {}
    for column in categorical_columns:
        categorical_distributions[column] = df[column].value_counts()

    # Création d'une instance de la classe avec toutes les informations
    return DataFrameExploration(
        column_types=column_types,
        unique_values=unique_values,
        numeric_stats=numeric_stats,
        missing_values=missing_values,
        categorical_distributions=categorical_distributions
    )

# Fonction pour convertir les colonnes de date en format datetime
def convert_dates(df, date_columns):
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # Conversion avec gestion des erreurs
    return df

# Fonction pour calculer la durée d'adhésion (en années)
def calculate_duration(df, adh_col, dem_col):
    if adh_col in df.columns and dem_col in df.columns:
        duration = (df[dem_col] - df[adh_col]).dt.days / 365.25
        return duration.describe()  # Retourne les statistiques descriptives
    return None


# Fonction de comparaison des résultats d'exploration
def compare_exploration_results(results1, results2):
    # Comparaison des types de colonnes
    common_columns = list(set(results1.column_types.index) & set(results2.column_types.index))  # Convertir en liste
    type_comparison = results1.column_types[common_columns] == results2.column_types[common_columns]

    # Comparaison des valeurs uniques
    unique_comparison = {}
    for column in common_columns:
        # Comparaison des ensembles de valeurs uniques pour gérer les longueurs différentes
        if isinstance(results1.unique_values[column], (list, np.ndarray)) and isinstance(results2.unique_values[column], (list, np.ndarray)):
            unique_comparison[column] = set(results1.unique_values[column]) == set(results2.unique_values[column])
        else:
            unique_comparison[column] = results1.unique_values[column] == results2.unique_values[column]

    # Comparaison des statistiques numériques
    numeric_comparison = results1.numeric_stats.equals(results2.numeric_stats)

    # Comparaison des valeurs manquantes
    missing_comparison = results1.missing_values.equals(results2.missing_values)

    # Comparaison des distributions catégorielles
    categorical_comparison = {}
    for column in common_columns:
        if column in results1.categorical_distributions and column in results2.categorical_distributions:
            categorical_comparison[column] = results1.categorical_distributions[column].equals(
                results2.categorical_distributions[column])
        else:
            categorical_comparison[column] = False  # Si la distribution n'existe pas dans une des deux tables

    # Retourne un DataFrame avec les résultats de la comparaison
    comparison_results = pd.DataFrame({
        'Column Types': type_comparison,
        'Unique Values': pd.Series(unique_comparison),
        'Numeric Stats': numeric_comparison,
        'Missing Values': missing_comparison,
        'Categorical Distributions': pd.Series(categorical_comparison)
    })

    return comparison_results

def common_points_resignation(dataname):
    # Lire les données de table1.csv
    data = pd.read_csv(dataname)

    # Filtrer uniquement les clients démissionnaires
    demissionnaires = data[data['CDDEM'].notna()]

    # Calculer le mode pour chaque colonne
    mode_values = demissionnaires.mode().iloc[0]  # Prendre seulement la première ligne du mode

    # Créer un DataFrame avec les noms de colonnes et leurs valeurs les plus fréquentes
    result = pd.DataFrame({
        'Column': mode_values.index,
        'Most Frequent Value': mode_values.values
    })

    # Afficher le résultat
    return result


def common_points_advanced(dataname):
    # Lire les données de table1.csv
    data = pd.read_csv(dataname)

    # Filtrer uniquement les clients démissionnaires
    demissionnaires = data[data['CDDEM'].notna()]

    # Calculer le mode pour chaque colonne (sauf ID) et ne prendre que la première ligne du mode
    mode_values = demissionnaires.drop(columns=['ID']).mode().iloc[0]
    

    # Pour chaque colonne, créer un tableau associatif avec à chaque valeur possible, sa fréquence associée
    frequency_df = {}
    for column in demissionnaires.columns:
        frequency_df[column] = demissionnaires[column].value_counts()

    # return le dataframe contenant les fréquences des valeurs pour chaque colonne
    return frequency_df




def common_points_plot(frequency_df):

    # mettre dans des fichiers txt les sorties des fréquences pour chaque colonne = un fichier txt différent
    for column in frequency_df:
        frequency_df[column].to_csv(f'../../Analyse/Mesures_frequences_valeurs/{column}_frequency.txt')
    
    # Créer un plot pour chaque colonne permettant de visualiser la distribution des valeurs
    for column in frequency_df:
        plt.figure(figsize=(10, 6))
        frequency_df[column].plot(kind='bar')
        plt.title(f'Distribution des valeurs pour la colonne {column}')
        plt.xlabel('Valeurs')
        plt.ylabel('Fréquence')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'../../Analyse/Plots_frequences_valeurs/{column}_frequency_plot.png')
        plt.close()
    

