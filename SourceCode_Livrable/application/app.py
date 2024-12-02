
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Configuration de l'application
st.set_page_config(page_title="Analyse Bancaire & Performance IA", layout="wide")

st.markdown("<h1 style='text-align: center;'>📊 Analyse des Données Bancaires & Performances des Modèles IA</h1>", unsafe_allow_html=True)

# Chargement des données fusionnées
csv_file_path = 'data/data_after_merging/table_merged_no_duplicates.csv'
if os.path.exists(csv_file_path):
    data = pd.read_csv(csv_file_path)

    # Section : Statistiques Clés
    st.subheader("🔍 Statistiques Clés")
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre Total de Sociétaires", len(data))
    col2.metric("Age Moyen des sociètaires (années)", f"{data['AGE'].mean():.1f} ans")
    col3.metric("Montant Moyen des Revenus", f"{data['MTREV'].mean():,.2f} €")

    # Section : Filtrage interactif
    st.subheader("📂 Filtrer les Données")
    sexe = st.multiselect("Filtrer par Sexe (CDSEXE)", options=data['CDSEXE'].unique())
    situation_fam = st.multiselect("Filtrer par Situation Familiale (CDSITFAM)", options=data['CDSITFAM'].unique())
    type_client = st.multiselect("Filtrer par Type de Client (CDCATCL)", options=data['CDCATCL'].unique())

    filtered_data = data.copy()
    if sexe:
        filtered_data = filtered_data[filtered_data['CDSEXE'].isin(sexe)]
    if situation_fam:
        filtered_data = filtered_data[filtered_data['CDSITFAM'].isin(situation_fam)]
    if type_client:
        filtered_data = filtered_data[filtered_data['CDCATCL'].isin(type_client)]

    st.dataframe(filtered_data)

    # Section : Graphiques
    st.subheader("📈 Visualisations")

    # Histogramme des tranches d'âge à l'adhésion
    fig1 = px.histogram(filtered_data, x='AGEAD', title="Répartition des Sociétaires par Âge à l'Adhésion")
    st.plotly_chart(fig1)

    # Graphique circulaire sur les motifs de démission
    if 'CDMOTDEM' in filtered_data.columns:
        fig2 = px.pie(filtered_data, names='CDMOTDEM', title="Répartition des Motifs de Démission")
        st.plotly_chart(fig2)

    # Histogramme des revenus des sociétaires
    fig3 = px.histogram(filtered_data, x='MTREV', title="Distribution des Revenus des Sociétaires", nbins=20)
    st.plotly_chart(fig3)

    # Histogramme du nombre d'enfants des sociétaires
    fig4 = px.histogram(filtered_data, x='NBENF', title="Répartition des Sociétaires selon le Nombre d'Enfants", nbins=6)
    st.plotly_chart(fig4)

    # Histogramme de la situation familiale des sociétaires
    fig5 = px.histogram(filtered_data, x='CDSITFAM', title="Répartition des Sociétaires par Situation Familiale")
    st.plotly_chart(fig5)

    # Age adésion de la durée d'adhésion
    fig6 = px.histogram(filtered_data, x='AGEAD', title="Age à l'adhésion des Sociétaires (en années)", nbins=10)
    st.plotly_chart(fig6)

    # Section spéciale : Graphiques de performance des modèles IA
    st.subheader("🤖 Performances des Modèles IA")

    graph_files = [f'fig/plot_predictions/performance_model_{i}.png' for i in range(1, 5)]
    for graph_file in graph_files:
        if os.path.exists(graph_file):
            st.image(graph_file, caption=f"Performance : {graph_file}", use_column_width=True)
else:
    st.error("Le fichier CSV de données fusionnées est introuvable.")
