# 🌐 Analyse de l’Indice Mondial de la Criminalité Organisée (2023)

Cette application interactive Streamlit permet d'explorer, comparer et prédire les niveaux de criminalité organisée à travers le monde, en utilisant des données issues de l'Indice Mondial de la Criminalité Organisée 2023.

## 🔍 Fonctionnalités

### 🧠 Modèle de Machine Learning
- Modèle : **Random Forest Regressor**
- But : Prédiction du score de criminalité
- Performance : **R² ≈ 0.95** sur les données de test
- Données normalisées, gestion des valeurs manquantes

### 📊 Tableau de bord interactif
- Carte choroplèthe mondiale des scores de criminalité
- Analyse régionale (boxplots, nuages de points)
- Comparaison entre pays (diagrammes radar)
- **Outil de prédiction dynamique** avec des sliders
- Visualisation des marchés criminels
- Filtrage par **continent**, **région** et **niveau de criminalité**
- Conception **responsive** (fonctionne sur mobile/tablette)

## 🗂 Données
Les données proviennent du rapport [Global Organized Crime Index 2023](https://ocindex.net/).  
L’analyse porte sur 193 pays selon trois piliers :
- **Marchés criminels**
- **Acteurs criminels**
- **Résilience des institutions**

## ⚙️ Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-utilisateur/votre-repo.git
cd votre-repo

# Installer les dépendances
pip install -r requirements.txt

# Lancer l’application
streamlit run dashboard.py
