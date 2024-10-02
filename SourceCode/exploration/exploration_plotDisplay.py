import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
data1 = pd.read_csv('../data/table1.csv')
data2 = pd.read_csv('../data/table2.csv')

# Configuration de style pour les graphiques
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# Fonction pour générer des graphiques pour les colonnes numériques
def plot_numeric_columns(df, table_name):
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    for column in numeric_columns:
        # Histogramme
        plt.figure()
        sns.histplot(df[column], kde=True, bins=30, color='blue')
        plt.title(f"Distribution de {column} - {table_name}")
        plt.xlabel(column)
        plt.ylabel("Fréquence")
        plt.grid(True)
        plt.savefig(f'../../Analyse/Plots_exploration/{table_name}_{column}_distribution.png')  # Sauvegarder l'image
        plt.close()  # Fermer la figure pour libérer de la mémoire

        # Box plot pour voir la plage des valeurs
        plt.figure()
        sns.boxplot(x=df[column])
        plt.title(f"Box plot de {column} - {table_name}")
        plt.grid(True)
        plt.savefig(f'../../Analyse/Plots_exploration/{table_name}_{column}_boxplot.png')
        plt.close()  # Fermer la figure pour libérer de la mémoire

# Fonction pour générer des graphiques pour les colonnes catégoriques
def plot_categorical_columns(df, table_name):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    for column in categorical_columns:
        plt.figure()
        sns.countplot(data=df, x=column, palette="Set2", legend=False)  # Supprimer le warning
        plt.title(f"Répartition des valeurs de {column} - {table_name}")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Nombre d'occurrences")
        plt.grid(True)
        plt.savefig(f'../../Analyse/Plots_exploration/{table_name}_{column}_countplot.png')  # Sauvegarder l'image
        plt.close()  # Fermer la figure pour libérer de la mémoire

# Fonction pour visualiser les valeurs manquantes
def plot_missing_values(df, table_name):
    plt.figure()
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title(f"Valeurs manquantes dans {table_name}")
    plt.savefig(f'../../Analyse/Plots_exploration/{table_name}_missing_values.png')  # Sauvegarder l'image
    plt.close()  # Fermer la figure pour libérer de la mémoire

# Visualisation des colonnes numériques pour Table 1
plot_numeric_columns(data1, "Table 1")

# Visualisation des colonnes catégoriques pour Table 1
plot_categorical_columns(data1, "Table 1")

# Visualisation des valeurs manquantes pour Table 1
plot_missing_values(data1, "Table 1")

# Répéter les visualisations pour Table 2
plot_numeric_columns(data2, "Table 2")
plot_categorical_columns(data2, "Table 2")
plot_missing_values(data2, "Table 2")
