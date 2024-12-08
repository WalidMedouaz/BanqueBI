
# Application BI

Ce projet consiste à explorer des données bancaires contenant à la fois des démissionnaires et des non démissionnaires. L'objectif final est de prédire quels clients sont les plus susceptibles de démissionner.




## Pré-Requis

Il y a un certain nombre de packages à installer que nous avons regroupé dans un fichier ``requirements.txt``.

Il faut d'abord lancer la commande ci-dessous.

```python
pip install -r requirements.txt
```

## Architecture

Les fonctionnalités sont réparties dans plusieurs fichiers, présents dans chaque dossier :
- `exploration` : fonctions pour explorer les données et générer les graphiques
- `nettoyage` : fonctions de nettoyage des données.
- `fusion` : fonctions pour fusionner les données (Outer Join).
- `recodage` : fonctions liées au recodage des variables.
- `prétraitement` : fonctions dédiées aux étapes de prétraitement des données et découpage (80%, 10%, 10%).
- `prédiction` : fonctions permettant de faire les prédictions basées sur les modèles Knn, SVM, Naive Bayes et Random Forest.

## Utilisation

Nous avons un fichier `main` principal qui va appeler l'ensemble des `main` de chaque fonctionnalité comme ci-dessous.

```python
function Main() {
  ...
}

Main()
```

##

Il suffira de lancer le `main` principal pour exécuter toutes les étapes comme ci-dessous.

```bash
python mainBI.py
```



## Collaboration/Tests
En ce qui concerne la manière dont il faut exécuter le script, nous avons un fichier `main` dans chaque dossier de notre architecture.

Cela nous permet de séparer les différentes étapes d'implémentation pour que chacun puisse travailler sur ses propres fonctionnalités. Cela nous permet aussi de faciliter les tests ou encore de faciliter l'identification et la résolution de bugs.

## Auteurs

- [@RechidiAbdelghani](https://github.com/abdelghanirechidi)
- [@SaccomanAlexis](https://github.com/AlexisSaccoman)
- [@MedouazWalid](https://github.com/WalidMedouaz)