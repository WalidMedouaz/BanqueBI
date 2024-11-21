## -------------------------------------------------------------------- IMPORT LIBS ------------------------------------------------------------------ ##
import pandas as pd

## ------------------------------------------------------------------- IMPORT FILES ------------------------------------------------------------------ ##
import functions_step.exploration_functions as ef
import functions_step.nettoyage_functions_v2 as nf

## -------------------------------------------------------------------- VARIABLES -------------------------------------------------------------------- ##
# ../fig/
# ../data/

path_folder_data = 'data/'
cleandDataFolderPath = "data/data_after_cleaning/"


## ------------------------------------------------------------------- EXPLORATION ------------------------------------------------------------------- ##

# Chargement des données
dataSample1 = pd.read_csv(path_folder_data + 'table1.csv')
dataSample2 = pd.read_csv(path_folder_data + 'table2.csv')

# Exploration des données
analysRes1 = ef.explore_dataframe(dataSample1)
analysRes2 = ef.explore_dataframe(dataSample2)

# comparaison des résultats
comparison = ef.compare_exploration_results(analysRes1, analysRes2)

# affichage de la comparaison des résultats d'exploration
print("#####################################################################################################################")
print("Comparaison des résultats d'exploration")
print(comparison)
print("#####################################################################################################################")
# Affichage des valeurs les plus représentées dans la table 1 et génération du plot affichant la variance cumulée des points communs
print("Affichage des points communs entre la liste des démissionnaires (table 1)")
cp = ef.common_points_advanced(path_folder_data + 'table1.csv')
# print(cp)
# appel de la fonction qui créera les plots/sorties permettant de visualiser les données pertinentes
print("Affichage des plots pour de distribution des valeurs")
#! ef.common_points_plot(cp)
print("#####################################################################################################################")
print("Distribution des valeurs")
resNum = ef.data_distribution(path_folder_data + 'table1.csv')
#! print(resNum)
print("#####################################################################################################################")


## -------------------------------------------------------------------- NETTOYAGE -------------------------------------------------------------------- ##


# Nettoyage des données
cleanedData1, cleanedData2 = nf.nettoyage([path_folder_data + 'table1.csv', path_folder_data + 'table2.csv'])


# Affichage des données nettoyées
print("#####################################################################################################################")
print(cleanedData1.head())
print("#####################################################################################################################")
print(cleanedData2.head())
print("#####################################################################################################################")
print("Sauvegarde des données nettoyées dans ../data_after_cleaning/")
# sauvegarder les fichiers nettoyés dans le dossier ../data_after_cleaning/
cleanedData1.to_csv(f"{cleandDataFolderPath}table1_cleaned.csv", index=False)
cleanedData2.to_csv(f"{cleandDataFolderPath}table2_cleaned.csv", index=False)
print("#####################################################################################################################")


## ---------------------------------------------------------------------- FUSION --------------------------------------------------------------------- ##

## --------------------------------------------------------------------- RECODAGE -------------------------------------------------------------------- ##

## ------------------------------------------------------------------ PRETRAITEMENT ------------------------------------------------------------------ ##

## -------------------------------------------------------------------- PREDICTION ------------------------------------------------------------------- ##