from fusion_functions import merge_and_replace, check_duplicates, delete_duplicates

# Exécution du processus de fusion, vérification et suppression des doublons
def main():
    # Fusion des tables et remplacement des valeurs par défaut
    outer_join_result = merge_and_replace()
    
    # Vérification des doublons dans le résultat de la jointure
    print("Vérification des doublons dans la table fusionnée :")
    check_duplicates(outer_join_result)
    
    # Suppression des doublons et sauvegarde du résultat final
    print("Suppression des doublons et sauvegarde de la table finale :")
    delete_duplicates(outer_join_result)


main()
