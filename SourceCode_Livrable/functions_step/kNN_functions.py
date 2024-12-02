from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns



def knn_prediction(X_train, X_test, y_train, y_test, k=5):
    # Initialiser le modèle kNN
    knn = KNeighborsClassifier(n_neighbors=k)
    # Entraîner le modèle
    knn.fit(X_train, y_train)
    # Prédire sur l'ensemble de test
    y_pred = knn.predict(X_test)
    # Évaluer le modèle
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for k={k}: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalisation de la matrice de confusion pour obtenir les pourcentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False)
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités terrain')
    plt.title(f'Matrice de confusion - kNN (k={k}) en %')
    plt.savefig("fig/plot_predictions/kNN/kNN_confusion.png")

    
    return accuracy
