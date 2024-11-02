import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import discretisation as disc

def normalisation(df):


	# Sélection des colonnes numériques
	numerical_cols = ['NBENF']

	# Normalisation Min-Max
	scaler_minmax = MinMaxScaler()
	df[numerical_cols] = scaler_minmax.fit_transform(df[numerical_cols])

	# Standardisation
	scaler_standard = StandardScaler()
	df[numerical_cols] = scaler_standard.fit_transform(df[numerical_cols])

	return df