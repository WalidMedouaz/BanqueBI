import pandas as pd
import exploration_functions as ef

# Chargement des données
dataSample1 = pd.read_csv('../data/table1.csv')
dataSample2 = pd.read_csv('../data/table2.csv')

# Exploration des données
analysRes1 = ef.explore_dataframe(dataSample1)
analysRes2 = ef.explore_dataframe(dataSample2)

# comparaison des résultats
comparison = ef.compare_exploration_results(analysRes1, analysRes2)

# affichage de la comparaison des résultats d'exploration
print("#####################################################################################################################")
print("Comparaison des résultats d'exploration:")
print(comparison)
print("#####################################################################################################################")
# Affichage des valeurs les plus représentées dans la table 1 et génération du plot affichant la variance cumulée des points communs
print("Affichage des points communs entre la liste des démissionnaires (table 1):")
cp = ef.common_points_advanced('../data/table1.csv')
print(cp)
# appel de la fonction qui créera les plots/sorties permettant de visualiser les données pertinentes
ef.common_points_plot(cp)
print("#####################################################################################################################")
