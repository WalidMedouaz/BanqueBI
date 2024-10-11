import pandas as pd
import nettoyage_functions_v2 as nf

# Nettoyage des données
cleanedData1, cleanedData2 = nf.nettoyage(['../data/table1.csv', '../data/table2.csv'])


# Affichage des données nettoyées
print("#####################################################################################################################")
print(cleanedData1.head())
print("#####################################################################################################################")
print(cleanedData2.head())
print("#####################################################################################################################")
print("Sauvegarde des données nettoyées dans ../data_after_cleaning/")
# sauvegarder les fichiers nettoyés dans le dossier ../data_after_cleaning/
cleandDataFolderPath = "../data_after_cleaning/"
cleanedData1.to_csv(f"{cleandDataFolderPath}table1_cleaned.csv", index=False)
cleanedData2.to_csv(f"{cleandDataFolderPath}table2_cleaned.csv", index=False)
print("#####################################################################################################################")