import pandas as pd

def discretisation():

	# Chargement des données
	df = pd.read_csv('../data_after_merging/table_merged_no_duplicates.csv')

	# Discrétisation de l'âge en tranches
	df['AGEAD_CAT'] = pd.cut(df['AGEAD'], bins=[18, 30, 50, 65, 100], labels=['Jeune', 'Adulte', 'Sénior', 'Aîné'])

	df['AGE_CAT'] = pd.cut(df['AGE'], bins=[18, 30, 50, 65, 100], labels=['Jeune', 'Adulte', 'Sénior', 'Aîné'])

	# Calcul des 4 quantiles pour la colonne MTREV sans compter les valeurs égales à 0
	quantiles_no_zeros = df[df['MTREV'] > 0]['MTREV'].quantile([0.25, 0.5, 0.75, 1.0])

	# Définir des intervalles manuels pour MTREV
	df['REV_CAT'] = pd.cut(df['MTREV'], bins=[0, quantiles_no_zeros[0.25], quantiles_no_zeros[0.5], quantiles_no_zeros[0.75], df['MTREV'].max()], labels=['Bas', 'Moyen', 'Élevé', 'Très élevé'])

	df.drop(columns=['MTREV', 'AGE', 'AGEAD'], inplace=True)

	return df