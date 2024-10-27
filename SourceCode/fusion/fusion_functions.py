import pandas as pd
import numpy as np

# Définition des chemins (fichiers en entrée)
file1_path = '../data_after_cleaning/table1_cleaned.csv'
file2_path = '../data_after_cleaning/table2_cleaned.csv'

# Définition des chemins (fichiers en sortie)
outer_join_file_path = '../data_after_merging/table_merged.csv'
outer_join_file_path_no_duplicates = '../data_after_merging/table_merged_no_duplicates.csv'

def merge_and_replace():

    # Chargement des tables
    table1 = pd.read_csv(file1_path)
    table2 = pd.read_csv(file2_path)

    # Jointure externe complète sur les colonnes spécifiques
    outer_join_result = pd.merge(
        table1, table2, how='outer', on=['CDSEXE', 'MTREV', 'NBENF', 'CDSITFAM', 'DTADH', 'CDTMT', 'CDCATCL', 'DTDEM', 'CDMOTDEM']
    )

    # Remplacement des valeurs absurdes (date et âge par défaut) par NaN
    outer_join_result['DTNAIS'] = outer_join_result['DTNAIS'].replace("1900-01-00", "NaN")
    outer_join_result.loc[outer_join_result['AGE'] == 107, 'AGE'] = "NaN"

    # Sauvegarde des modifications dans un fichier CSV après l'outer join
    outer_join_result.to_csv(outer_join_file_path, index=False)
    return outer_join_result

def check_duplicates(outer_join_result):

    # Vérification des doublons après la jointure
    duplicates = outer_join_result[outer_join_result.duplicated(keep=False)]

    # Nombre de doublons
    nb_doublons = len(duplicates)
    print("Nombre de doublons :", nb_doublons)

    # Affichage des doublons
    print(duplicates)

def delete_duplicates(outer_join_result):

    # Suppression des doublons à partir du résultat de l'outer join
    outer_join_result_no_duplicates = outer_join_result.drop_duplicates()

    # Vérification du nombre de lignes avant et après suppression des duplicates
    print("Nombre de lignes avant suppression des duplicates :", len(outer_join_result))
    print("Nombre de lignes après suppression des duplicates :", len(outer_join_result_no_duplicates))

    # Sauvegarde de la fusion finale sans doublons
    outer_join_result_no_duplicates.to_csv(outer_join_file_path_no_duplicates, index=False)
