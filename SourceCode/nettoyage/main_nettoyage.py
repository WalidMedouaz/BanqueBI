import pandas as pd
import nettoyage_functions as nf

# Nettoyage des données
cleanedData1, cleanedData2 = nf.nettoyage(['../data/table1.csv', '../data/table2.csv'])

# Affichage des données nettoyées
print("#####################################################################################################################")
print(cleanedData1)
print("#####################################################################################################################")
print(cleanedData2)
print("#####################################################################################################################")