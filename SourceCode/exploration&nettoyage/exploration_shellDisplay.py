import pandas as pd

# Chargement des données
data1 = pd.read_csv('../data/table1.csv')
data2 = pd.read_csv('../data/table2.csv')

# Fonction pour explorer les colonnes d'un dataframe
def explore_dataframe(df, table_name):
    print(f"\nExploration des données pour {table_name} :")
    
    # Types des colonnes
    print("\nTypes des colonnes :")
    print(df.dtypes)
    
    # Valeurs uniques pour chaque colonne ou la plage si plus de 5 valeurs
    print("\nValeurs possibles pour chaque colonne :")
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:  # Si c'est une colonne numérique
            print(f"{column}: Plage de valeurs ({df[column].min()} - {df[column].max()})")
        else:
            unique_values = df[column].unique()
            if len(unique_values) <= 5:
                print(f"{column}: {unique_values}")
            else:
                print(f"{column}: Trop de valeurs uniques pour les afficher ({len(unique_values)} valeurs)")
    
    # Calcul des moyennes pour les colonnes numériques
    print("\nStatistiques sur les colonnes numériques :")
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if not numeric_columns.empty:
        print(df[numeric_columns].describe())  # Moyennes, médiane, min, max, etc.
    else:
        print("Pas de colonnes numériques dans cette table.")

    # Comptage des valeurs manquantes
    print("\nValeurs manquantes par colonne :")
    print(df.isnull().sum())

    # Distribution des colonnes catégoriques
    print("\nDistribution des colonnes catégoriques :")
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for column in categorical_columns:
        print(f"\n{column} :")
        print(df[column].value_counts())

# Exploration des deux tables
explore_dataframe(data1, "Table 1")
explore_dataframe(data2, "Table 2")

# Conversion des colonnes de date en format datetime
def convert_dates(df, date_columns):
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # Conversion et gestion des erreurs
    return df

# Appliquer la conversion des dates
data1 = convert_dates(data1, ['DTADH', 'DTDEM'])
data2 = convert_dates(data2, ['DTADH', 'DTDEM', 'DTNAIS'])

# Calcul des durées d'adhésion pour table 1
if 'DTADH' in data1.columns and 'DTDEM' in data1.columns:
    data1['DureeAdhesion'] = (data1['DTDEM'] - data1['DTADH']).dt.days / 365.25
    print("\nDurée d'adhésion (en années) pour table 1 :")
    print(data1['DureeAdhesion'].describe())

# Calcul des durées d'adhésion pour table 2
if 'DTADH' in data2.columns and 'DTDEM' in data2.columns:
    data2['DureeAdhesion'] = (data2['DTDEM'] - data2['DTADH']).dt.days / 365.25
    print("\nDurée d'adhésion (en années) pour table 2 :")
    print(data2['DureeAdhesion'].describe())
