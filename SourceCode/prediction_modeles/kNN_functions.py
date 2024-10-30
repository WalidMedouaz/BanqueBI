from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


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
    return accuracy
