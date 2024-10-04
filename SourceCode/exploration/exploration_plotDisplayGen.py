import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
data1 = pd.read_csv('../data/table1.csv')
data2 = pd.read_csv('../data/table2.csv')

# Configuration de style pour les graphiques
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [15, 10]  # Augmenter la taille des figures pour les regrouper

# Fonction pour regrouper les distributions des colonnes numériques
def plot_numeric_columns_combined(df, table_name):
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    num_cols = len(numeric_columns)
    
    # Créer des sous-graphes
    fig, axes = plt.subplots(nrows=(num_cols // 2) + (num_cols % 2), ncols=2, figsize=(16, num_cols * 2.5))
    axes = axes.flatten()
    
    for i, column in enumerate(numeric_columns):
        sns.histplot(df[column], kde=True, bins=30, color='blue', ax=axes[i])
        axes[i].set_title(f"Distribution de {column}")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Fréquence")
    
    plt.suptitle(f"Distributions des colonnes numériques - {table_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'../../Analyse/Plots_exploration_gen/{table_name}_numeric_distributions_combined.png')
    plt.close()

# Fonction pour regrouper les box plots des colonnes numériques
def plot_numeric_boxplots_combined(df, table_name):
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=df[numeric_columns], orient="h")
    plt.title(f"Box plots des colonnes numériques - {table_name}")
    plt.grid(True)
    plt.savefig(f'../../Analyse/Plots_exploration_gen/{table_name}_numeric_boxplots_combined.png')
    plt.close()

# Fonction pour regrouper les countplots des colonnes catégoriques
def plot_categorical_columns_combined(df, table_name):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    num_cols = len(categorical_columns)
    
    fig, axes = plt.subplots(nrows=(num_cols // 2) + (num_cols % 2), ncols=2, figsize=(16, num_cols * 3))
    axes = axes.flatten()
    
    for i, column in enumerate(categorical_columns):
        sns.countplot(data=df, x=column, palette="Set2", legend=False, ax=axes[i])
        axes[i].set_title(f"Répartition de {column}")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha="right")
        axes[i].set_ylabel("Nombre d'occurrences")
    
    plt.suptitle(f"Répartition des colonnes catégoriques - {table_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'../../Analyse/Plots_exploration_gen/{table_name}_categorical_countplots_combined.png')
    plt.close()

# Fonction pour visualiser les valeurs manquantes
def plot_missing_values(df, table_name):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title(f"Valeurs manquantes dans {table_name}")
    plt.savefig(f'../../Analyse/Plots_exploration_gen/{table_name}_missing_values.png')
    plt.close()

# Visualisation des colonnes numériques pour Table 1
plot_numeric_columns_combined(data1, "Table 1")

# Visualisation des box plots des colonnes numériques pour Table 1
plot_numeric_boxplots_combined(data1, "Table 1")

# Visualisation des colonnes catégoriques pour Table 1
plot_categorical_columns_combined(data1, "Table 1")

# Visualisation des valeurs manquantes pour Table 1
plot_missing_values(data1, "Table 1")

# Répéter les visualisations pour Table 2
plot_numeric_columns_combined(data2, "Table 2")
plot_numeric_boxplots_combined(data2, "Table 2")
plot_categorical_columns_combined(data2, "Table 2")
plot_missing_values(data2, "Table 2")
