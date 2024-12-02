from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC


def svm_prediction(X_train, X_test, y_train, y_test, kernel='linear'):
    print(f"Début de SVM avec noyau={kernel}...")
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    print("Entraînement terminé.")

    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with kernel='{kernel}': {accuracy:.2f}")
    
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False)
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités terrain')
    plt.title(f'Matrice de confusion - SVM (kernel={kernel}) en %')
    plt.savefig("fig/plot_predictions/SVM/SVM_" + kernel + "_confusion.png")

    return accuracy