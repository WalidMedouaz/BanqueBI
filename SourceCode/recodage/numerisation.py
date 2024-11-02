import pandas as pd
import discretisation as disc
import normalisation as norm

recodage_path = "../data_after_recodage/table_recodage.csv"

def numerisation(df):

	# Numérisation avec One-Hot Encoding
	df = pd.get_dummies(df, columns=['CDSEXE', 'CDSITFAM', 'CDTMT', 'CDMOTDEM', 'CDCATCL', 'AGEAD_CAT', 'AGE_CAT', 'REV_CAT'], drop_first=True)

	df.to_csv(recodage_path, index=False)
	