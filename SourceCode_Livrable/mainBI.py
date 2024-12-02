## -------------------------------------------------------------------- IMPORT LIBS ------------------------------------------------------------------ ##
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import plotly.express as px
import os

## ------------------------------------------------------------------- IMPORT FILES ------------------------------------------------------------------ ##
import functions_step.exploration_functions as ef
import functions_step.nettoyage_functions_v2 as nf
import functions_step.fusion_functions as ff
import functions_step.discretisation as disc
import functions_step.numerisation as num
import functions_step.normalisation as norm
import functions_step.pretraitement_functions as pf
import functions_step.kNN_functions as kf
import functions_step.SVM_functions as sF
import functions_step.Bayes_functions as bF
import functions_step.Random_Forest_functions as rF

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
print("Sauvegarde des données nettoyées dans ../data/data_after_cleaning/")
# sauvegarder les fichiers nettoyés dans le dossier ../data_after_cleaning/
cleanedData1.to_csv(f"{cleandDataFolderPath}table1_cleaned.csv", index=False)
cleanedData2.to_csv(f"{cleandDataFolderPath}table2_cleaned.csv", index=False)
print("#####################################################################################################################")


## ---------------------------------------------------------------------- FUSION --------------------------------------------------------------------- ##

# Fusion des tables et remplacement des valeurs par défaut
outer_join_result = ff.merge_and_replace()

print("#####################################################################################################################")
    
# Vérification des doublons dans le résultat de la jointure
print("Vérification des doublons dans la table fusionnée :")
ff.check_duplicates(outer_join_result)

print("#####################################################################################################################")
    
# Suppression des doublons et sauvegarde du résultat final
print("Suppression des doublons et sauvegarde de la table finale :")
ff.delete_duplicates(outer_join_result)

print("#####################################################################################################################")

## --------------------------------------------------------------------- RECODAGE -------------------------------------------------------------------- ##
print("#####################################################################################################################")
print("Recodage des données")
num.numerisation(norm.normalisation(disc.discretisation()))
print("#####################################################################################################################")

## ------------------------------------------------------------------ PRETRAITEMENT ------------------------------------------------------------------ ##

print("#####################################################################################################################")
data = pd.read_csv("data/data_after_merging/table_merged_no_duplicates.csv")

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
dataReady.to_csv("data/data_after_pretraitement/data_after_pretraitement.csv", index=False)
X_train.to_csv("data/data_after_pretraitement/X_train.csv", index=False)
X_test.to_csv("data/data_after_pretraitement/X_test.csv", index=False)
y_train.to_csv("data/data_after_pretraitement/y_train.csv", index=False)
y_test.to_csv("data/data_after_pretraitement/y_test.csv", index=False)
X_val.to_csv("data/data_after_pretraitement/X_val.csv", index=False)
y_val.to_csv("data/data_after_pretraitement/y_val.csv", index=False)

print(" ")
print("Data saved successfully")
print("###################################################################")

## -------------------------------------------------------------------- PREDICTION ------------------------------------------------------------------- ##



# Charger les données prétraitées
X_train = pd.read_csv("data/data_after_pretraitement/X_train.csv")
X_test = pd.read_csv("data/data_after_pretraitement/X_test.csv")
y_train = pd.read_csv("data/data_after_pretraitement/y_train.csv").values.ravel()
y_test = pd.read_csv("data/data_after_pretraitement/y_test.csv").values.ravel()
x_val = pd.read_csv("data/data_after_pretraitement/X_val.csv")
y_val = pd.read_csv("data/data_after_pretraitement/y_val.csv").values.ravel()



print("###################################################################") # ----------------------------------------------------------------- # 
# kNN
# Liste pour stocker les précisions
accuracies = []

# Tester plusieurs valeurs de k
for k in range(1, 20):
    acc = kf.knn_prediction(X_train, X_test, y_train, y_test, k)
    accuracies.append(acc)

# Visualiser les précisions en fonction des valeurs de k
plt.plot(range(1, 20), accuracies, marker='o')
plt.xlabel('Valeur de k')
plt.ylabel('Précision')
plt.title('Précision du modèle kNN en fonction de k')
plt.savefig("fig/plot_predictions/kNN/kNN_accuracy.png")
plt.close()

# Afficher la valeur de k qui donne la meilleure précision et afficher la précision correspondate avec la moyenne
best_k = np.argmax(accuracies) + 1
print(f"Meilleure valeur de k: {best_k}")
print(f"Précision correspondante: {accuracies[best_k - 1]:.2f}")
print(f"Précision moyenne: {np.mean(accuracies):.2f}")

print("###################################################################") # ----------------------------------------------------------------- # 
# Bayes

print("Début de bayes ...")

# Obtenir la précision pour Naive Bayes
bayes_accuracy = bF.bayes_prediction(X_train, X_test, y_train, y_test)

# Générer un plot pour la précision
plt.figure(figsize=(6, 4))
plt.bar(['Naive Bayes'], [bayes_accuracy], color='orange')
plt.xlabel('Modèle')
plt.ylabel('Précision')
plt.title('Précision du modèle Naive Bayes')
plt.savefig("fig/plot_predictions/Bayes/bayes_accuracy.png")
plt.close()

print("###################################################################") # ----------------------------------------------------------------- # 
# SVM
# Modèle : Plutot lent et pas optimisé pour les grands jeux de données
# Définition d'une liste de noyaux pour tester différents modèles SVM
kernels = ['rbf', 'poly', 'sigmoid']  # On peut ajouter d'autres noyaux si besoin : ['linear', 'poly', 'rbf', 'sigmoid']
#                             

# Mettre en commentaire ça si tu veux test sur le jeu de données initial (prend plus de temps, modifie sur les inputs en bas (enlever le small))
# X_train_small = X_train.sample(frac=0.1, random_state=42)  # 10% des données
# y_train_small = y_train[:len(X_train_small)]

svm_accuracies = []

# Boucle pour tester chaque noyau
for kernel in kernels:
    # Appel de la fonction svm_prediction et obtenir la précision pour chaque noyau
    acc = sF.svm_prediction(X_train, X_test, y_train, y_test, kernel)
    # Ajout de la précision obtenue à la liste des précisions
    svm_accuracies.append(acc)

# Affichage de la précision moyenne pour les différents noyaux SVM testés
print(f"Précision moyenne pour SVM: {np.mean(svm_accuracies):.2f}")

# Plot de la précision pour chaque noyau
plt.figure(figsize=(10, 6))
plt.plot(kernels, svm_accuracies, marker='o', linestyle='-')
plt.xlabel('Type de noyau SVM')
plt.ylabel('Précision')
plt.title('Précision du modèle SVM pour différents noyaux')
plt.grid(True)
plt.savefig("fig/plot_predictions/SVM/svm_accuracy_per_kernel.png")
#plt.show()


print("###################################################################") # ----------------------------------------------------------------- # 
# Random Forest

# Tester le modèle Random Forest avec différents nombres d'arbres (estimators)
n_estimators_list = [20, 40, 60, 80, 100, 120, 140, 160]
rf_accuracies = []

for n_estimators in n_estimators_list:
    acc = rF.random_forest_prediction(X_train, X_test, y_train, y_test, n_estimators)
    rf_accuracies.append(acc)

# Plot de la précision en fonction du nombre d'estimateurs
plt.figure(figsize=(8, 6))
plt.plot(n_estimators_list, rf_accuracies, marker='o', linestyle='-', color='green')
plt.xlabel('Nombre d\'arbres (n_estimators)')
plt.ylabel('Précision')
plt.title('Précision du modèle Random Forest')
plt.grid(True)
plt.savefig("fig/plot_predictions/Random_Forest/random_forest_accuracy.png")
plt.close()

print(f"Meilleure précision obtenue : {max(rf_accuracies):.2f} avec {n_estimators_list[rf_accuracies.index(max(rf_accuracies))]} arbres.")

print("###################################################################") # ----------------------------------------------------------------- # 

## ---------------------------------------------------------------- MODEL PERF DISPLAY --------------------------------------------------------------- ##

# génére un graphique permettant de comparer les performances des différents modèles
plt.figure(figsize=(10, 6))
plt.bar(['kNN', 'Naive Bayes', 'SVM', 'Random Forest'], [np.mean(accuracies), bayes_accuracy, np.mean(svm_accuracies), max(rf_accuracies)])
plt.xlabel('Modèle')
plt.ylabel('Précision')
plt.title('Comparaison des performances des modèles')
plt.grid(True)
plt.savefig("fig/plot_predictions/Model_Performance/model_performance.png")
plt.close()


## -------------------------------------------------------------------- DASHBOARD -------------------------------------------------------------------- ##

# lancer le fichier app.py grâce à OS
os.system('streamlit run ./application/app.py')