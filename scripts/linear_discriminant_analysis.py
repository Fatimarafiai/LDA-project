import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("📊 LINEAR DISCRIMINANT ANALYSIS - CLASSIFICATION AVEC LABELS")
print("="*70)

# ============================================
# 1️⃣ CHARGER LES DONNÉES AVEC LABELS
# ============================================
print("\n[1/6] 📂 Chargement des données avec LABELS...")
try:
    df = pd.read_csv('data/lda_dataset.csv')
    print(f"✅ {len(df)} documents chargés")
    print(f"\n📋 Classes (LABELS):")
    for i, cls in enumerate(df['label'].unique(), 1):
        count = (df['label'] == cls).sum()
        print(f"   {i}. {cls} : {count} documents")
    print(f"\nDistribution des classes:")
    print(df['label'].value_counts())
except FileNotFoundError:
    print("❌ Erreur: Fichier 'data/lda_dataset.csv' non trouvé")
    exit()

# ============================================
# 2️⃣ PRÉPARATION DES DONNÉES
# ============================================
print("\n[2/6] 🔤 Vectorization TF-IDF...")
vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label'].values

print(f"✅ Vocabulaire: {X.shape[1]} features")
print(f"✅ Documents: {X.shape[0]}")
print(f"✅ Classes: {len(np.unique(y))}\n")

# ============================================
# 3️⃣ SPLIT TRAIN/TEST AVEC LABELS
# ============================================
print("[3/6] 📊 Split Train/Test (75/25) avec LABELS...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    random_state=42, 
    stratify=y
)

print(f"✅ Training set: {len(X_train)} documents")
print(f"   Distribution: {dict(pd.Series(y_train).value_counts())}")
print(f"\n✅ Test set: {len(X_test)} documents")
print(f"   Distribution: {dict(pd.Series(y_test).value_counts())}\n")

# ============================================
# 4️⃣ ENTRAÎNER LDA AVEC LABELS
# ============================================
print("[4/6] 🤖 Entraînement LDA avec LABELS...")
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train, y_train)

print(f"✅ LDA entraîné sur {len(X_train)} documents")
print(f"✅ Variance expliquée par composante:")
for i, var in enumerate(lda.explained_variance_ratio_):
    print(f"   Composante {i+1}: {var:.2%}")
print()

# ============================================
# 5️⃣ PRÉDICTIONS SUR TEST SET
# ============================================
print("[5/6] 🎯 Prédictions sur TEST SET...")

y_pred = lda.predict(X_test)
y_pred_proba = lda.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ ACCURACY SUR TEST SET: {accuracy:.2%}\n")

print("📋 CLASSIFICATION REPORT (TEST SET):")
print("="*70)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\n🔥 CONFUSION MATRIX (TEST SET):")
print(cm)

# ============================================
# 6️⃣ PRÉDICTIONS SUR TOUS LES DOCUMENTS
# ============================================
print("\n[6/6] 📈 Prédictions sur TOUS les documents...")

X_all_pred = lda.predict(X)
X_all_pred_proba = lda.predict_proba(X)

df['predicted_class'] = X_all_pred
df['confidence'] = np.max(X_all_pred_proba, axis=1)
df['correct'] = df['label'] == df['predicted_class']

print(f"\n✅ Résumé des prédictions:")
print(f"   Documents corrects: {df['correct'].sum()} / {len(df)}")
print(f"   Accuracy globale: {df['correct'].mean():.2%}")
print(f"\n✅ Top 10 prédictions avec confiance:")
print(df[['id', 'label', 'predicted_class', 'confidence', 'correct']].head(10))

# ============================================
# 🎨 VISUALIZATIONS
# ============================================

print("\n[VISUALIZATIONS] Création des graphiques...")

import os
if not os.path.exists('output'):
    os.makedirs('output')

# 1️⃣ Confusion Matrix
print("1️⃣ Confusion Matrix...")
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=lda.classes_, 
            yticklabels=lda.classes_,
            cbar_kws={'label': 'Count'},
            linewidths=2,
            linecolor='black')
plt.title('Confusion Matrix - LDA Classification (TEST SET)', fontsize=14, fontweight='bold')
plt.ylabel('True Class (Réalité)', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Class (Prédiction)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('output/01_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: output/01_confusion_matrix.png")
plt.close()

# 2️⃣ LDA Projection 2D
print("2️⃣ LDA Projection 2D...")
X_lda = lda.transform(X)

plt.figure(figsize=(12, 8))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
markers = ['o', 's', '^', 'D']

for i, label in enumerate(lda.classes_):
    indices = y == label
    plt.scatter(X_lda[indices, 0], X_lda[indices, 1], 
               label=label, alpha=0.7, s=150, color=colors[i % len(colors)], 
               edgecolors='black', linewidth=2, marker=markers[i % len(markers)],
               zorder=3)

plt.xlabel(f'LDA 1 ({lda.explained_variance_ratio_[0]:.1%} de variance)', 
          fontsize=12, fontweight='bold')
plt.ylabel(f'LDA 2 ({lda.explained_variance_ratio_[1]:.1%} de variance)', 
          fontsize=12, fontweight='bold')
plt.title('LDA Projection 2D - Séparation des Classes', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='best', title='Classes')
plt.grid(alpha=0.3, linestyle='--', zorder=0)
plt.tight_layout()
plt.savefig('output/02_lda_projection_2d.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: output/02_lda_projection_2d.png")
plt.close()

# 3️⃣ Accuracy par Classe
print("3️⃣ Accuracy par Classe...")
plt.figure(figsize=(10, 6))
class_accuracies = []
for cls in lda.classes_:
    mask = y_test == cls
    if mask.sum() > 0:
        acc = accuracy_score(y_test[mask], y_pred[mask])
    else:
        acc = 0
    class_accuracies.append(acc)

bars = plt.bar(range(len(lda.classes_)), class_accuracies, 
              color=colors[:len(lda.classes_)], 
              edgecolor='black', linewidth=2)
plt.xticks(range(len(lda.classes_)), lda.classes_, rotation=15, ha='right')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title('Accuracy par Classe - TEST SET', fontsize=14, fontweight='bold')
plt.ylim(0, 1.1)

for bar, acc in zip(bars, class_accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/03_accuracy_per_class.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: output/03_accuracy_per_class.png")
plt.close()

# 4️⃣ Confiance des prédictions
print("4️⃣ Distribution de confiance...")
plt.figure(figsize=(10, 6))
plt.hist(df['confidence'], bins=20, color='#4ECDC4', edgecolor='black', linewidth=1.5, alpha=0.7)
plt.xlabel('Confiance (Probabilité)', fontsize=12, fontweight='bold')
plt.ylabel('Nombre de documents', fontsize=12, fontweight='bold')
plt.title('Distribution de Confiance des Prédictions', fontsize=14, fontweight='bold')
plt.axvline(df['confidence'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Moyenne: {df["confidence"].mean():.2%}')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('output/04_confidence_distribution.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: output/04_confidence_distribution.png")
plt.close()

# 5️⃣ Résultats corrects vs incorrects
print("5️⃣ Résultats corrects vs incorrects...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

correct_count = df['correct'].sum()
incorrect_count = len(df) - correct_count
ax1.pie([correct_count, incorrect_count], 
       labels=[f'Correct\n({correct_count})', f'Incorrect\n({incorrect_count})'],
       colors=['#4ECDC4', '#FF6B6B'],
       autopct='%1.1f%%',
       startangle=90,
       textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('Résultats de Classification', fontsize=12, fontweight='bold')

correct_per_class = []
for cls in lda.classes_:
    mask = df['label'] == cls
    correct = (df[mask]['correct']).sum()
    total = mask.sum()
    correct_per_class.append(correct / total if total > 0 else 0)

ax2.bar(range(len(lda.classes_)), correct_per_class, 
       color=colors[:len(lda.classes_)], edgecolor='black', linewidth=1.5)
ax2.set_xticks(range(len(lda.classes_)))
ax2.set_xticklabels(lda.classes_, rotation=15, ha='right')
ax2.set_ylabel('Taux de Réussite', fontsize=12, fontweight='bold')
ax2.set_title('Taux de Réussite par Classe', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output/05_correct_vs_incorrect.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: output/05_correct_vs_incorrect.png")
plt.close()

# ============================================
# 💾 SAUVEGARDER LES RÉSULTATS
# ============================================
print("\n[RÉSULTATS] Sauvegarde des données...")

df.to_csv('output/lda_all_predictions.csv', index=False)
print("   ✅ Saved: output/lda_all_predictions.csv")

test_results = pd.DataFrame({
    'True Class': y_test,
    'Predicted Class': y_pred,
    'Correct': y_test == y_pred
})
test_results.to_csv('output/lda_test_set_predictions.csv', index=False)
print("   ✅ Saved: output/lda_test_set_predictions.csv")

summary = {
    'Métrique': [
        'Accuracy (Test Set)',
        'Total Documents',
        'Training Size',
        'Test Size',
        'Nombre de Classes',
        'Features',
        'Documents Correct',
        'Documents Incorrect'
    ],
    'Valeur': [
        f'{accuracy:.2%}',
        len(df),
        len(X_train),
        len(X_test),
        len(lda.classes_),
        X.shape[1],
        f'{df["correct"].sum()}',
        f'{(~df["correct"]).sum()}'
    ]
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv('output/lda_summary.csv', index=False)
print("   ✅ Saved: output/lda_summary.csv")

classes_info = pd.DataFrame({
    'Class': lda.classes_,
    'Total': [np.sum(y == cls) for cls in lda.classes_],
    'Training': [np.sum(y_train == cls) for cls in lda.classes_],
    'Testing': [np.sum(y_test == cls) for cls in lda.classes_]
})
classes_info.to_csv('output/lda_classes_info.csv', index=False)
print("   ✅ Saved: output/lda_classes_info.csv")

print("\n" + "="*70)
print("🎉 LINEAR DISCRIMINANT ANALYSIS - CLASSIFICATION TERMINÉE !")
print("="*70)
print(f"\n✅ RÉSUMÉ FINAL:")
print(f"   • Accuracy sur TEST SET: {accuracy:.2%}")
print(f"   • Total documents: {len(df)}")
print(f"   • Classes: {', '.join(lda.classes_)}")
print(f"   • Fichiers générés: 5 PNG + 4 CSV")
print(f"\n📊 Fichiers disponibles dans 'output/'")
print("="*70)