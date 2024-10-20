import pandas as pd

# Suppression des colonnes inutiles pour table1
file1_path = '../../donnees_banque/table1.csv'
cleaned_file1_path = '../../donnees_banque/apres_fusion/table1.csv'

columns_to_remove = ['ID', 'ANNEEDEM', 'AGEDEM', 'ADH', 'RANGAGEAD', 'RANGAGEDEM', 'RANGADH', 'RANGDEM']

table1 = pd.read_csv(file1_path)

table1_cleaned = table1.drop(columns=columns_to_remove)

table1_cleaned.to_csv(cleaned_file1_path, index=False)

# Suppression des colonnes inutiles pour table2
file2_path = '../../donnees_banque/table2.csv'
cleaned_file2_path = '../../donnees_banque/apres_fusion/table2.csv'

columns_to_remove = ['BPADH']

table2 = pd.read_csv(file2_path)

table2_cleaned = table2.drop(columns=columns_to_remove)

table2_cleaned.to_csv(cleaned_file2_path, index=False)

# Outer Join (Jointure externe complète)
outer_join_result = pd.merge(table1_cleaned, table2_cleaned, how='outer', on=['CDSEXE', 'MTREV', 'NBENF', 'CDSITFAM', 'DTADH', 'CDTMT', 'CDCATCL', 'DTDEM', 'CDMOTDEM'])

# Left Join (Jointure gauche)
left_join_result = pd.merge(table1_cleaned, table2_cleaned, how='left', on=['CDSEXE', 'MTREV', 'NBENF', 'CDSITFAM', 'DTADH', 'CDTMT', 'CDCATCL', 'DTDEM', 'CDMOTDEM'])

outer_join_file_path = '../../donnees_banque/apres_fusion/outer_join_result.csv'
left_join_file_path = '../../donnees_banque/apres_fusion/left_join_result.csv'

outer_join_result.to_csv(outer_join_file_path, index=False)
left_join_result.to_csv(left_join_file_path, index=False)
