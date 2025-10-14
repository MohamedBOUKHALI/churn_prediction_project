# Projet Data Science : Prédiction du Désabonnement Client (Churn)

## Table des matières
- [Contexte et Objectifs](#contexte-et-objectifs)
- [Architecture du Projet](#architecture-du-projet)
- [Pré-analyse des Données (EDA)](#pré-analyse-des-données-eda)
- [Préparation des Données & Modélisation](#préparation-des-données--modélisation)
- [Évaluation des Modèles](#évaluation-des-modèles)
- [Tests Unitaires](#tests-unitaires)
- [Instructions de Lancement](#instructions-de-lancement)

---

## Contexte et Objectifs

### Contexte Métier
L'entreprise de télécommunications fait face à un **taux de désabonnement élevé (Churn)** impactant directement ses revenus. Le churn est un problème stratégique influencé par les contrats, les services souscrits et les historiques de paiement. L'absence de solution IA ciblée rend les campagnes de fidélisation inefficaces.

### Objectif Principal
Développer un **pipeline complet de Machine Learning supervisé** pour identifier les clients ayant la probabilité la plus élevée de se désabonner (Churn = Yes), afin de permettre à l'équipe Marketing de lancer des campagnes de fidélisation ciblées et proactives.

---

## Architecture du Projet

Ce projet est organisé autour des meilleures pratiques pour garantir la reproductibilité et la qualité du code.

| Fichier / Outil | Rôle dans le Pipeline | Exigences Adressées |
|------------------|----------------------|---------------------|
| `notebook.ipynb` | EDA et Visualisation. Analyse initiale des données, identification des problèmes (NaN, déséquilibre), présentation des résultats d'évaluation finale (courbes ROC et PR). | EDA, Visualisation, Résultats Intermédiaires |
| `pipeline.py` | Cœur du Pipeline ML. Contient toutes les fonctions de pré-traitement (nettoyage, encodage), de split, d'entraînement et d'évaluation des modèles. | Préparation des données, Entraînement, Automation |
| `test_pipeline.py` | Tests Unitaires. Validation de la robustesse des fonctions critiques (chargement, nettoyage, split). | Tests Unitaires (Robustesse) |
| GitHub / Git | Versionnement du Code. Suivi des modifications et organisation. | Git / Versionnement |

---

## Pré-analyse des Données (EDA)

L'exploration initiale (`notebook.ipynb`) a permis de structurer les étapes de préparation des données :

| Observation | Implication pour le Modèle | Résolution (`pipeline.py`) |
|-------------|----------------------------|----------------------------|
| **Déséquilibre de Classe** | La variable cible (Churn) est déséquilibrée (≈73% No, ≈27% Yes). ⟹ Le Recall et le F1-Score sont les métriques prioritaires. | Priorité aux métriques adaptées |
| **TotalCharges (Valeurs Manquantes)** | Contient des chaînes vides `' '` nécessitant une conversion en numérique. | ⟹ Géré dans `clean_data()` par imputation par la moyenne. |
| **Encodage Catégoriel** | Toutes les variables object doivent être transformées en numérique. | ⟹ LabelEncoder est utilisé pour l'encodage de toutes les colonnes object. |

---

##  Préparation des Données & Modélisation

### A. Méthode de Pré-traitement & Pipeline

Le pipeline séquentiel appliqué aux données inclut :

1. **Nettoyage** (`clean_data`) : Suppression de la clé `customerID` et gestion des valeurs manquantes.
2. **Encodage** (`encode_categorical`) : Transformation des variables catégorielles.
3. **Normalisation** : Les données numériques sont mises à l'échelle (impliqué par l'utilisation de LogisticRegression et souvent complété par MinMaxScaler dans le code source).
4. **Split** (`split_data`) : Séparation en ensembles d'entraînement et de test.

### B. Modèles Entraînés

Deux modèles de classification supervisée ont été entraînés (dans `entrain_model()` de `pipeline.py`) :

- **LogisticRegression** : Modèle linéaire simple.
- **RandomForestClassifier** : Modèle d'ensemble non-linéaire.

---

## Évaluation des Modèles

L'évaluation est basée sur les résultats obtenus sur l'ensemble de test, en privilégiant le **Recall** pour l'équipe marketing.

### A. Tableau Comparatif des Métriques (Résultats Réels)

| Métrique | Poids pour le Churn | Logistic Regression | Random Forest |
|----------|---------------------|---------------------|---------------|
| **Recall** (Taux de Vrais Positifs) | Primaire (Identification des clients partants) | **0.5581** | 0.5127 |
| **F1-Score** | Équilibre Précision/Recall | **0.6062** | 0.5647 |
| **Accuracy** | Taux de prédiction correcte global | **0.8183** | 0.8020 |
| **ROC AUC Score** | Performance globale de classification | **0.8455** | 0.8260 |

### B. Interprétation et Justification du Modèle Retenu

**Le modèle Logistic Regression est retenu pour la mise en production.**

**Justifications :**

- **Priorité au Recall (55.81%)** : La Régression Logistique a un Recall supérieur (55.81% vs 51.27% pour RF). Dans le cadre du Churn, l'erreur la plus coûteuse est le **Faux Négatif** (ne pas identifier un client qui part). Le modèle Logistique, même s'il est plus simple, permet d'identifier un plus grand nombre de clients à risque, ce qui est l'exigence fondamentale de l'équipe marketing.

- **Meilleur F1-Score et ROC AUC** : Le modèle Logistique démontre également un meilleur équilibre (F1-Score de 0.6062) et une meilleure capacité de discrimination globale (ROC AUC de 0.8455), confirmant sa supériorité pour ce jeu de données et cette configuration.

- **Simplicité de Déploiement** : En tant que modèle linéaire, il est plus facile à comprendre, à expliquer (interprétabilité) et à maintenir en production pour une première solution IA.

---

##  Tests Unitaires

Le fichier `test_pipeline.py` utilise la librairie **pytest** pour garantir l'intégrité des données à travers les étapes critiques.

| Fonction Testée | Description | Statut |
|-----------------|-------------|--------|
| `test_split_data()` | Validation de la cohérence dimensionnelle des ensembles X et y après la séparation Train/Test. | ✅ RÉUSSI |

---

## Instructions de Lancement

### Prérequis
Assurez-vous d'avoir **Python 3.x** et les dépendances listées dans `requirements.txt`.

### Installation des Dépendances


# Créez et activez un environnement virtuel 
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installez toutes les bibliothèques requises
pip install -r requirements.txt


### Exécution du Pipeline et Entraînement


# Exécute le pipeline complet (nettoyage, split, entraînement, évaluation)
python pipeline.py


### Lancement des Tests Unitaires


# Valide la propreté et la cohérence des fonctions de préparation
pytest test_pipeline.py

## Livrables

- ✅ Pipeline ML complet et automatisé
- ✅ Notebook d'analyse exploratoire
- ✅ Tests unitaires validés
- ✅ Documentation technique complète
- ✅ Versionnement Git

---

## Équipe & Contact

Pour toute question concernant ce projet, veuillez me contacter sur l'email : ayoub.motei@gmail.com .

---
