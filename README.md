Linear Discriminant Analysis (LDA) - Classification Project
#Description
Projet de Classification supervisée avec Linear Discriminant Analysis (LDA).
Classifie automatiquement des documents en 4 catégories en utilisant des LABELS.
# Objectif
Entraîner un modèle LDA qui classe les documents en 4 catégories :
- Machine Learning - Algorithmes, réseaux de neurones, deep learning
- Software Development - Développement, coding, frameworks
- Data Analysis - Analyse statistique, visualisation, mining
- Architecture - Conception système, patterns, design
# Données
Total: 24 documents avec LABELS
Distribution:
- Machine Learning: 6 documents
- Software Development: 6 documents
- Data Analysis: 6 documents
- Architecture: 6 documents
Format: CSV (id, text, label)
Deux versions disponibles :
- lda_dataset.csv - Données originales
- lda_dataset_perturbed.csv - Données perturbées pour démonstration
Installation et Utilisation
# Etape 1: Cloner le repository
git clone https://github.com/Fatimarafiai/LDA-project.git
cd LDA-project
#Etape 2: Installer les dépendances
pip install -r requirements.txt
# Etape 3: Créer le dossier output
mkdir output
# Etape 4: Lancer le script LDA
python scripts/linear_discriminant_analysis.py
# Résultats Attendus
# Performance
Accuracy sur TEST SET: 95-100%
Tous les documents correctement classifiés
Séparation des classes EXCELLENTE
# Fichiers Générés
01_AVANT_LDA_PERTURBE_MELANGE.png - Données PERTURBEES avant LDA (Classes MELANGEES)
02_APRES_LDA_SEPARE_ORGANISE.png - Données ORGANISEES après LDA (Classes SEPAREES)
03_COMPARAISON_AVANT_APRES_LDA.png - Comparaison côte à côte AVANT/APRES
04_confusion_matrix.png - Matrice de confusion (TEST SET)
05_improvement_score.png - Score d'amélioration avec LDA
lda_all_predictions.csv - Prédictions pour tous les documents
results_summary.csv - Résumé des métriques
# AVANT vs APRES LDA
# Transformation des Données
# AVANT LDA (Données PERTURBEES)
État: CHAOTIQUES, MELANGEES ET PERTURBEES
Classes: COMPLETEMENT CONFONDUES
Score de séparation: TRES FAIBLE (Silhouette < 0)
Capacité de classification: IMPOSSIBLE
Visualisation: 01_AVANT_LDA_PERTURBE_MELANGE.png
# APRES LDA (Données ORGANISEES)
État: ORGANISEES, GROUPEES ET SEPAREES
Classes: CLAIREMENT DISTINCTES
Score de séparation: EXCELLENT (Silhouette > 0.9)
Capacité de classification: PARFAITE
Visualisation: 02_APRES_LDA_SEPARE_ORGANISE.png
# Comparaison Visuelle
Voir: 03_COMPARAISON_AVANT_APRES_LDA.png
Le graphe montre clairement :
- A GAUCHE: Les 4 classes MELANGEES (avant LDA)
- A DROITE: Les 4 classes SEPAREES (après LDA)
# Structure du Projet

LDA-project/
├── data/
│   ├── lda_dataset.csv (24 documents avec LABELS)
│   └── lda_dataset_perturbed.csv (Données perturbées)
├── scripts/
│   ├── linear_discriminant_analysis.py (Script principal LDA)
│   ├── visualize_before_after_lda.py (Visualisations AVANT/APRES)
│   └── visualize_chaotic_to_organized.py (Transformation CHAOS vers ORDRE)
├── output/ (Résultats auto-créé)
│   ├── 01_AVANT_LDA_PERTURBE_MELANGE.png
│   ├── 02_APRES_LDA_SEPARE_ORGANISE.png
│   ├── 03_COMPARAISON_AVANT_APRES_LDA.png
│   ├── 04_confusion_matrix.png
│   ├── 05_improvement_score.png
│   ├── lda_all_predictions.csv
│   └── results_summary.csv
├── README.md (Ce fichier)
├── requirements.txt (Dépendances)
└── .gitignore (Fichiers à ignorer)
# Technologies Utilisées
- Python 3.8+
- scikit-learn - Machine Learning (LDA)
- pandas - Manipulation de données
- matplotlib - Visualisation graphique
- seaborn - Graphiques statistiques
- numpy - Calculs numériques
- # Qu'est-ce que LDA (Linear Discriminant Analysis) ?
Linear Discriminant Analysis est un algorithme de classification supervisée qui :
# Caractéristiques
- Sépare les classes avec une limite linéaire
- Réduit la dimensionnalité des données (50 features vers 2D)
- Maximise la séparation entre classes
- Minimise la variance intra-classe
- Gère bien les données perturbées et mélangées
# Fonctionnement
1. Reçoit des documents avec des LABELS
2. Analyse la distribution de chaque classe
3. Entraîne sur 75% des données (TRAIN SET)
4. Teste sur 25% des données (TEST SET)
5. Prédit la classe des nouveaux documents
6. Transforme les données en 2D pour visualisation
# Exemple de Transformation
AVANT LDA :
- 50 dimensions (50 features)
- Classes CONFONDUES
- Score Silhouette : -0.5 (MAUVAIS)
APRES LDA :
- 2 dimensions (2 composantes)
- Classes SEPAREES
- Score Silhouette : 0.95 (EXCELLENT)
AMELIORATION : +1.45
# Résultats Métriques

Accuracy TEST SET: 95-100%
Score AVANT LDA: -0.5 (Perturbé)
Score APRES LDA: 0.95 (Séparé)
Amélioration: +1.45
Total Documents: 24
Nombre de Classes: 4
Features: 50

# Auteur
Fatima Rafiai
Date: Avril 2026
Dernière mise à jour: 2026-04-29

# Contact et Support
Pour des questions ou des suggestions, n'hésitez pas à ouvrir une issue sur GitHub.

LDA Transform Complex Data into Clear Classifications!
