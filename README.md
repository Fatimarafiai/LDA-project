 Linear Discriminant Analysis (LDA) - Classification Project

 Description
Projet de **Classification supervisée** avec Linear Discriminant Analysis (LDA).

**Classifie automatiquement des documents en 4 catégories** en utilisant des LABELS.

## Objectif
Entraîner un modèle LDA qui classe les documents en 4 catégories :
- **Machine Learning** - Algorithmes, réseaux de neurones, deep learning
- **Software Development** - Développement, coding, frameworks
- **Data Analysis** - Analyse statistique, visualisation, mining
- **Architecture** - Conception système, patterns, design

## Données

```
Total: 24 documents avec LABELS
Distribution:
├─ Machine Learning: 6 documents
├─ Software Development: 6 documents
├─ Data Analysis: 6 documents
└─ Architecture: 6 documents

Format: CSV (id, text, label)
```

## Installation & Utilisation

### Étape 1: Cloner le repository
```bash
git clone https://github.com/USERNAME/lda-project.git
cd lda-project
```

### Étape 2: Installer les dépendances
```bash
pip install -r requirements.txt
```

### Étape 3: Créer le dossier output
```bash
mkdir output
```

### Étape 4: Lancer le script LDA
```bash
python scripts/linear_discriminant_analysis.py
```

## 📈 Résultats Attendus

### Performance
```
✅ Accuracy sur TEST SET: ~100%
✅ Tous les documents classifiés correctement
```

### Fichiers Générés

| Fichier | Description |
|---------|-------------|
| `01_confusion_matrix.png` | Matrice de confusion (TEST SET) |
| `02_lda_projection_2d.png` | Projection 2D des classes |
| `03_accuracy_per_class.png` | Précision par classe |
| `04_confidence_distribution.png` | Distribution de confiance |
| `05_correct_vs_incorrect.png` | Résultats corrects vs incorrects |
| `lda_all_predictions.csv` | Prédictions pour tous les documents |
| `lda_test_set_predictions.csv` | Résultats du test set |
| `lda_summary.csv` | Résumé des métriques |
| `lda_classes_info.csv` | Information par classe |

## Structure du Projet

```
lda-project/
├── data/
│   └── lda_dataset.csv                    # 24 documents avec LABELS
├── scripts/
│   └── linear_discriminant_analysis.py    # Script principal
├── output/                                # Résultats (auto-créé)
│   ├── 01_confusion_matrix.png
│   ├── 02_lda_projection_2d.png
│   ├── 03_accuracy_per_class.png
│   ├── 04_confidence_distribution.png
│   ├── 05_correct_vs_incorrect.png
│   ├── lda_all_predictions.csv
│   ├── lda_test_set_predictions.csv
│   ├── lda_summary.csv
│   └── lda_classes_info.csv
├── README.md                              # Ce fichier
├── requirements.txt                       # Dépendances
└── .gitignore                             # Fichiers à ignorer
```

##  Technologies Utilisées

- **Python** 3.8+
- **scikit-learn** - Machine Learning (LDA)
- **pandas** - Manipulation de données
- **matplotlib** - Visualisation
- **seaborn** - Graphiques statistiques
- **numpy** - Calculs numériques

## 🎓 Qu'est-ce que LDA (Linear Discriminant Analysis) ?

**Linear Discriminant Analysis** est un algorithme de **classification supervisée** qui :

### Caractéristiques
- Sépare les classes avec une **limite linéaire**
- Réduit la **dimensionnalité** des données
- **Maximise** la séparation entre classes
- **Minimise** la variance intra-classe

### Fonctionnement
1. **Reçoit** des documents avec des LABELS
2. **Entraîne** sur 75% des données (TRAIN SET)
3. **Teste** sur 25% des données (TEST SET)
4. **Prédit** la classe des nouveaux documents

## Auteur

**Fatima**  
Date: Avril 2026

---

**Happy Classifying! 🚀**
