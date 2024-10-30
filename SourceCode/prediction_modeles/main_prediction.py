
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

import kNN_functions as kf


# Charger les données prétraitées
X_train = pd.read_csv("../data_after_pretraitement/X_train.csv")
X_test = pd.read_csv("../data_after_pretraitement/X_test.csv")
y_train = pd.read_csv("../data_after_pretraitement/y_train.csv")
y_test = pd.read_csv("../data_after_pretraitement/y_test.csv")
x_val = pd.read_csv("../data_after_pretraitement/X_val.csv")
y_val = pd.read_csv("../data_after_pretraitement/y_val.csv")


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
print("###################################################################") # ----------------------------------------------------------------- # 
# SVM
print("###################################################################") # ----------------------------------------------------------------- # 
# Tree
print("###################################################################") # ----------------------------------------------------------------- # 