
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pretraitement_functions as pf


data = pd.read_csv("../data_after_merging/table_merged_no_duplicates.csv")

print(data.info())
print("###################################################################")
data_preprocessed = pf.convert(data)
print(data_preprocessed.info())

print("###################################################################")
dataReady = data_preprocessed.copy()
X = dataReady.drop(columns=['dem_bool'])
y = dataReady['dem_bool']

# Diviser les données en ensembles : train (80%), test (10%) et validation (10%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Afficher les dimensions des données et les infos sur les données
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# On enregistre les données prétraitées dans un fichier csv (../data_after_pretraitement/data_after_pretraitement.csv) ainsi que les données d'entrainement, de test et de vlidation
dataReady.to_csv("../data_after_pretraitement/data_after_pretraitement.csv", index=False)
X_train.to_csv("../data_after_pretraitement/X_train.csv", index=False)
X_test.to_csv("../data_after_pretraitement/X_test.csv", index=False)
y_train.to_csv("../data_after_pretraitement/y_train.csv", index=False)
y_test.to_csv("../data_after_pretraitement/y_test.csv", index=False)
X_val.to_csv("../data_after_pretraitement/X_val.csv", index=False)
y_val.to_csv("../data_after_pretraitement/y_val.csv", index=False)

print(" ")
print("Data saved successfully")
print("###################################################################")
