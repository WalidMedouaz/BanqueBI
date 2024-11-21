from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# def svm_prediction(X_train, X_test, y_train, y_test, kernel='linear'):
#     print(f"Début de SVM avec noyau={kernel}...")  # Début de l'entraînement
#     svm = SVC(kernel=kernel)
    
#     # Entraîner le modèle
#     svm.fit(X_train, y_train)
#     print("Entraînement terminé.")  # Après entraînement
    
#     # Prédire sur l'ensemble de test
#     y_pred = svm.predict(X_test)
#     print("Prédiction terminée.")  # Après prédiction
    
#     # Calculer la précision
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy with kernel='{kernel}': {accuracy:.2f}")
    
#     return accuracy

# Fonction test sur 1000 itération en max iter (fonction du haut fait tout les calculs (prend plus de temps))

def svm_prediction(X_train, X_test, y_train, y_test, kernel='linear'):
    print(f"Début de SVM avec noyau={kernel}...")
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    print("Entraînement terminé.")

    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with kernel='{kernel}': {accuracy:.2f}")
    
    return accuracy