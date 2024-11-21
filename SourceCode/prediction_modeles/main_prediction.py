
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

import kNN_functions as kf
import SVM_functions as sF
import Bayes_functions as bF
import Random_Forest_functions as rF


# Charger les données prétraitées
X_train = pd.read_csv("../data_after_pretraitement/X_train.csv")
X_test = pd.read_csv("../data_after_pretraitement/X_test.csv")
y_train = pd.read_csv("../data_after_pretraitement/y_train.csv").values.ravel()
y_test = pd.read_csv("../data_after_pretraitement/y_test.csv").values.ravel()
x_val = pd.read_csv("../data_after_pretraitement/X_val.csv")
y_val = pd.read_csv("../data_after_pretraitement/y_val.csv").values.ravel()



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
plt.savefig("../../Analyse/plot_predictions/kNN/kNN_accuracy.png")
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
plt.savefig("../../Analyse/plot_predictions/Bayes/bayes_accuracy.png")
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
plt.savefig("../../Analyse/plot_predictions/SVM/svm_accuracy_per_kernel.png")
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
plt.savefig("../../Analyse/plot_predictions/Random_Forest/random_forest_accuracy.png")
plt.close()

print(f"Meilleure précision obtenue : {max(rf_accuracies):.2f} avec {n_estimators_list[rf_accuracies.index(max(rf_accuracies))]} arbres.")

print("###################################################################") # ----------------------------------------------------------------- # 