import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.preprocessing import StandardScaler

# ⚠️ Ceci doit être AVANT tout appel Streamlit
st.set_page_config(page_title="Indice mondial de la criminalité organisée", layout="wide")

# Chargement des données et du modèle
@st.cache_data
def charger_donnees():
    df = pd.read_excel('global_oc_index.xlsx', sheet_name='2023_dataset')
    return df

@st.cache_resource
def charger_modele():
    return joblib.load('crime_index_model.pkl')

df = charger_donnees()
modele = charger_modele()

# Titre principal
st.title("Analyse de l'indice mondial de la criminalité organisée Par Saliou DIEDHIOU")

# Filtres dans la barre latérale
st.sidebar.header("Filtres")
continent_selectionne = st.sidebar.multiselect(
    "Sélectionner un ou plusieurs continents",
    options=df['Continent'].unique(),
    default=df['Continent'].unique()
)

region_selectionnee = st.sidebar.multiselect(
    "Sélectionner une ou plusieurs régions",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

score_min, score_max = st.sidebar.slider(
    "Plage de scores de criminalité",
    min_value=float(df['Criminality avg,'].min()),
    max_value=float(df['Criminality avg,'].max()),
    value=(float(df['Criminality avg,'].min()), float(df['Criminality avg,'].max()))
)

# Filtrage des données
df_filtre = df[
    (df['Continent'].isin(continent_selectionne)) &
    (df['Region'].isin(region_selectionnee)) &
    (df['Criminality avg,'] >= score_min) &
    (df['Criminality avg,'] <= score_max)
]

# Tableau de bord principal
onglet1, onglet2, onglet3, onglet4 = st.tabs([
    "Vue globale", "Analyse régionale", "Comparaison entre pays", "Outil de prédiction"
])

with onglet1:
    st.header("Vue globale de la criminalité organisée")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.choropleth(
            df_filtre,
            locations="Country",
            locationmode="country names",
            color="Criminality avg,",
            hover_name="Country",
            hover_data=["Region", "Criminal markets avg,", "Resilience avg,"],
            color_continuous_scale="reds",
            title="Carte mondiale des scores de criminalité"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pays_haut = df_filtre.nlargest(10, 'Criminality avg,')
        pays_bas = df_filtre.nsmallest(10, 'Criminality avg,')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=pays_haut['Country'],
            x=pays_haut['Criminality avg,'],
            name='Plus forte criminalité',
            orientation='h',
            marker_color='crimson'
        ))
        fig.add_trace(go.Bar(
            y=pays_bas['Country'],
            x=pays_bas['Criminality avg,'],
            name='Plus faible criminalité',
            orientation='h',
            marker_color='lightgreen'
        ))
        fig.update_layout(
            title="Top et Flop 10 des pays selon le score de criminalité",
            barmode='group',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

with onglet2:
    st.header("Analyse régionale")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            df_filtre,
            x="Region",
            y="Criminality avg,",
            color="Continent",
            title="Distribution des scores par région"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df_filtre,
            x="Criminality avg,",
            y="Resilience avg,",
            color="Continent",
            size="Criminal markets avg,",
            hover_name="Country",
            title="Lien entre criminalité et résilience"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Moyennes par marché criminel
    marches_criminels = [
        'Human trafficking', 'Human smuggling', 'Arms trafficking',
        'Flora crimes', 'Fauna crimes', 'Non-renewable resource crimes',
        'Heroin trade', 'Cocaine trade', 'Cannabis trade',
        'Synthetic drug trade', 'Cyber-dependent crimes',
        'Financial crimes', 'Trade in counterfeit goods',
        'Illicit trade in excisable goods', 'Extortion and protection racketeering'
    ]

    moyennes = df_filtre[marches_criminels].mean().sort_values(ascending=False)
    fig = px.bar(
        moyennes,
        orientation='h',
        title="Score moyen par marché criminel",
        labels={'value': 'Score moyen', 'index': 'Marché criminel'}
    )
    st.plotly_chart(fig, use_container_width=True)

with onglet3:
    st.header("Comparaison entre pays")

    pays_selectionnes = st.multiselect(
        "Choisir des pays à comparer",
        options=df['Country'].unique(),
        default=["Mexico", "Senegal", "Afghanistan", "Myanmar"]
    )

    if pays_selectionnes:
        df_compare = df_filtre[df_filtre['Country'].isin(pays_selectionnes)]
        categories = [
            'Criminality avg,', 'Criminal markets avg,', 
            'Criminal actors avg,', 'Resilience avg,'
        ]

        fig = go.Figure()
        for pays in pays_selectionnes:
            donnees_pays = df_compare[df_compare['Country'] == pays]
            fig.add_trace(go.Scatterpolar(
                r=donnees_pays[categories].values[0],
                theta=categories,
                fill='toself',
                name=pays
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title='Comparaison radar entre pays'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Données détaillées")
        st.dataframe(
            df_compare.set_index('Country')[categories].T,
            use_container_width=True
        )

with onglet4:
    st.header("Outil de prédiction du score de criminalité")

    st.write("""
    Cet outil prédit le score de criminalité d’un pays en fonction de divers indicateurs.
    Ajustez les curseurs pour observer l’effet de chaque variable.
    """)

    col1, col2 = st.columns(2)

    with col1:
        criminal_markets = st.slider("Marchés criminels (moyenne)", 0.0, 10.0, 5.0, 0.1)
        human_trafficking = st.slider("Trafic d'êtres humains", 0.0, 10.0, 5.0, 0.1)
        arms_trafficking = st.slider("Trafic d'armes", 0.0, 10.0, 5.0, 0.1)
        drug_trade = st.slider("Commerce de drogues (moyenne)", 0.0, 10.0, 5.0, 0.1)
        financial_crimes = st.slider("Crimes financiers", 0.0, 10.0, 5.0, 0.1)

    with col2:
        criminal_actors = st.slider("Acteurs criminels (moyenne)", 0.0, 10.0, 5.0, 0.1)
        mafia_groups = st.slider("Groupes mafieux", 0.0, 10.0, 5.0, 0.1)
        state_actors = st.slider("Acteurs étatiques corrompus", 0.0, 10.0, 5.0, 0.1)
        resilience = st.slider("Résilience (moyenne)", 0.0, 10.0, 5.0, 0.1)
        law_enforcement = st.slider("Forces de l'ordre", 0.0, 10.0, 5.0, 0.1)

    input_data = {
        'Criminal markets avg,': criminal_markets,
        'Human trafficking': human_trafficking,
        'Human smuggling': (human_trafficking + arms_trafficking)/2,
        'Arms trafficking': arms_trafficking,
        'Flora crimes': (criminal_markets + human_trafficking)/2,
        'Fauna crimes': (criminal_markets + human_trafficking)/2,
        'Non-renewable resource crimes': financial_crimes,
        'Heroin trade': drug_trade,
        'Cocaine trade': drug_trade,
        'Cannabis trade': drug_trade,
        'Synthetic drug trade': drug_trade,
        'Cyber-dependent crimes': financial_crimes,
        'Financial crimes': financial_crimes,
        'Trade in counterfeit goods': financial_crimes,
        'Illicit trade in excisable goods': financial_crimes,
        'Extortion and protection racketeering': mafia_groups,
        'Criminal actors avg,': criminal_actors,
        'Mafia-style groups': mafia_groups,
        'Criminal networks': criminal_actors,
        'State-embedded actors': state_actors,
        'Foreign actors': criminal_actors,
        'Private sector actors': criminal_actors,
        'Resilience avg,': resilience,
        'Political leadership and governance': resilience,
        'Government transparency and accountability': resilience,
        'International cooperation': resilience,
        'National policies and laws': resilience,
        'Judicial system and detention': law_enforcement,
        'Law enforcement': law_enforcement,
        'Territorial integrity': resilience,
        'Anti-money laundering': law_enforcement,
        'Economic regulatory capacity': resilience,
        'Victim and witness support': resilience,
        'Prevention': resilience,
        'Non-state actors': resilience
    }

    input_df = pd.DataFrame([input_data])

    if st.button("Prédire le score de criminalité"):
        prediction = modele.predict(input_df)[0]

        st.metric("Score prédit", f"{prediction:.2f}")

        st.subheader("Interprétation")
        if prediction < 3:
            st.success("Niveau faible – Institutions fortes, faible présence criminelle")
        elif prediction < 6:
            st.warning("Niveau modéré – Défis présents mais maîtrisables")
        else:
            st.error("Niveau élevé – Présence significative du crime organisé")

        st.subheader("Facteurs clés influents")
        facteurs_importants = {
            'Marchés criminels': criminal_markets,
            'Acteurs criminels': criminal_actors,
            'Commerce de drogues': drug_trade,
            'Crimes financiers': financial_crimes,
            'Résilience': resilience
        }

        fig = px.bar(
            x=list(facteurs_importants.values()),
            y=list(facteurs_importants.keys()),
            orientation='h',
            title="Contribution des principaux facteurs"
        )
        st.plotly_chart(fig, use_container_width=True)

# Pied de page
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Source des données :** Indice mondial du crime organisé 2023  
**Méthodologie :** Évaluation de 193 pays selon 3 piliers : marchés criminels, acteurs criminels, résilience.
""")
