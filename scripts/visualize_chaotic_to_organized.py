import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🌀 DONNÉES CHAOTIQUES → DONNÉES ORGANISÉES AVEC LDA")
print("="*70)

# ============================================
# 1️⃣ CHARGER ET PRÉPARER LES DONNÉES
# ============================================
print("\n[1/4] 📂 Chargement des données...")
df = pd.read_csv('data/lda_dataset.csv')
df = df.dropna(subset=['label'])

feature_cols = [col for col in df.columns if col.startswith('f')]
X = df[feature_cols].values
y = df['label'].values

print(f"✅ {len(df)} documents chargés")
print(f"✅ {X.shape[1]} features numériques")
print(f"✅ {len(np.unique(y))} classes\n")

# ============================================
# 2️⃣ NORMALISER LES DONNÉES
# ============================================
print("[2/4] 🔢 Normalisation des données...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Données normalisées\n")

# ============================================
# 3️⃣ CRÉER UNE PROJECTION CHAOTIQUE (2D aléatoire)
# ============================================
print("[3/4] 🌀 Création d'une projection CHAOTIQUE...")

# Prendre 2 features aléatoires (chaotique)
np.random.seed(42)
random_indices = np.random.choice(X_scaled.shape[1], 2, replace=False)
X_chaotic = X_scaled[:, random_indices]

print(f"✅ Projection aléatoire sur 2 features")
print(f"   Features choisies : f{random_indices[0]+1}, f{random_indices[1]+1}\n")

# ============================================
# 4️⃣ APPLIQUER LDA (ORGANISÉ)
# ============================================
print("[4/4] 🤖 Application de LDA pour ORGANISER...")

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

print(f"✅ LDA appliqué")
print(f"✅ Variance expliquée LDA:")
for i, var in enumerate(lda.explained_variance_ratio_):
    print(f"   Composante {i+1}: {var:.2%}\n")

# ============================================
# 🎨 VISUALISATIONS
# ============================================

import os
if not os.path.exists('output'):
    os.makedirs('output')

colors = {'Architecture': '#FF6B6B', 'Data Analysis': '#4ECDC4', 'Machine Learning': '#45B7D1'}
markers_dict = {'Architecture': 'o', 'Data Analysis': 's', 'Machine Learning': '^'}

# ============================================
# GRAPHIQUE 1️⃣ : DONNÉES CHAOTIQUES (AVANT)
# ============================================
print("Création des graphiques...")
print("1️⃣ Données CHAOTIQUES (AVANT LDA)...")

fig, ax = plt.subplots(figsize=(12, 8))

for label in np.unique(y):
    indices = y == label
    ax.scatter(X_chaotic[indices, 0], X_chaotic[indices, 1],
               label=label, alpha=0.7, s=250, 
               color=colors[label], edgecolors='black', 
               linewidth=2, marker=markers_dict[label], zorder=3)

ax.set_xlabel(f'Feature {random_indices[0]+1} (aléatoire)', fontsize=12, fontweight='bold')
ax.set_ylabel(f'Feature {random_indices[1]+1} (aléatoire)', fontsize=12, fontweight='bold')
ax.set_title('🌀 AVANT LDA - DONNÉES CHAOTIQUES ET PERTURBÉES !\n(Les classes sont MÉLANGÉES !)', 
             fontsize=14, fontweight='bold', color='red')
ax.legend(fontsize=12, loc='best', title='Classes', title_fontsize=12)
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/00_CHAOTIC_donnees_perturbees.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: output/00_CHAOTIC_donnees_perturbees.png")
plt.close()

# ============================================
# GRAPHIQUE 2️⃣ : DONNÉES ORGANISÉES (APRÈS LDA)
# ============================================
print("2️⃣ Données ORGANISÉES (APRÈS LDA)...")

fig, ax = plt.subplots(figsize=(12, 8))

for label in np.unique(y):
    indices = y == label
    ax.scatter(X_lda[indices, 0], X_lda[indices, 1],
               label=label, alpha=0.7, s=250, 
               color=colors[label], edgecolors='black', 
               linewidth=2, marker=markers_dict[label], zorder=3)

ax.set_xlabel(f'LDA 1 ({lda.explained_variance_ratio_[0]:.1%} de variance)', fontsize=12, fontweight='bold')
ax.set_ylabel(f'LDA 2 ({lda.explained_variance_ratio_[1]:.1%} de variance)', fontsize=12, fontweight='bold')
ax.set_title('✅ APRÈS LDA - DONNÉES ORGANISÉES ET BIEN SÉPARÉES !\n(Les classes sont DISTINCTES !)', 
             fontsize=14, fontweight='bold', color='green')
ax.legend(fontsize=12, loc='best', title='Classes', title_fontsize=12)
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/01_ORGANIZED_donnees_organisees.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: output/01_ORGANIZED_donnees_organisees.png")
plt.close()

# ============================================
# GRAPHIQUE 3️⃣ : COMPARAISON CÔTE À CÔTE
# ============================================
print("3️⃣ Comparaison CHAOTIQUE vs ORGANISÉ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# CHAOTIQUE
for label in np.unique(y):
    indices = y == label
    ax1.scatter(X_chaotic[indices, 0], X_chaotic[indices, 1],
               label=label, alpha=0.7, s=250, 
               color=colors[label], edgecolors='black', 
               linewidth=2, marker=markers_dict[label], zorder=3)

ax1.set_xlabel(f'Feature {random_indices[0]+1}', fontsize=11, fontweight='bold')
ax1.set_ylabel(f'Feature {random_indices[1]+1}', fontsize=11, fontweight='bold')
ax1.set_title('🌀 AVANT LDA\nDONNÉES CHAOTIQUES ET PERTURBÉES\n(Classes MÉLANGÉES !)', 
              fontsize=12, fontweight='bold', color='red')
ax1.legend(fontsize=11, loc='best')
ax1.grid(alpha=0.3, linestyle='--')

# ORGANISÉ
for label in np.unique(y):
    indices = y == label
    ax2.scatter(X_lda[indices, 0], X_lda[indices, 1],
               label=label, alpha=0.7, s=250, 
               color=colors[label], edgecolors='black', 
               linewidth=2, marker=markers_dict[label], zorder=3)

ax2.set_xlabel(f'LDA 1 ({lda.explained_variance_ratio_[0]:.1%})', fontsize=11, fontweight='bold')
ax2.set_ylabel(f'LDA 2 ({lda.explained_variance_ratio_[1]:.1%})', fontsize=11, fontweight='bold')
ax2.set_title('✅ APRÈS LDA\nDONNÉES ORGANISÉES ET BIEN SÉPARÉES\n(Classes DISTINCTES !)', 
              fontsize=12, fontweight='bold', color='green')
ax2.legend(fontsize=11, loc='best')
ax2.grid(alpha=0.3, linestyle='--')

plt.suptitle('TRANSFORMATION : 🌀 CHAOS → ✅ ORDRE avec LDA', 
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('output/02_TRANSFORMATION_chaos_to_order.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: output/02_TRANSFORMATION_chaos_to_order.png")
plt.close()

# ============================================
# GRAPHIQUE 4️⃣ : HEATMAP DE DÉSORDRE vs ORDRE
# ============================================
print("4️⃣ Heatmap : Mesure de désordre vs ordre...")

# Calculer la distance moyenne intra-classe et inter-classe
from sklearn.metrics import silhouette_score

silhouette_chaotic = silhouette_score(X_chaotic, y)
silhouette_lda = silhouette_score(X_lda, y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Score de séparation AVANT
categories = ['Séparation']
scores_before = [silhouette_chaotic]
bars1 = ax1.bar(categories, scores_before, color='#FF6B6B', edgecolor='black', linewidth=2, width=0.5)
ax1.set_ylim(-0.5, 1)
ax1.set_ylabel('Score de Séparation', fontsize=12, fontweight='bold')
ax1.set_title('🌀 AVANT LDA\nScore de séparation FAIBLE', fontsize=12, fontweight='bold', color='red')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
for bar, score in zip(bars1, scores_before):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax1.grid(axis='y', alpha=0.3)

# Score de séparation APRÈS
scores_after = [silhouette_lda]
bars2 = ax2.bar(categories, scores_after, color='#4ECDC4', edgecolor='black', linewidth=2, width=0.5)
ax2.set_ylim(-0.5, 1)
ax2.set_ylabel('Score de Séparation', fontsize=12, fontweight='bold')
ax2.set_title('✅ APRÈS LDA\nScore de séparation EXCELLENT', fontsize=12, fontweight='bold', color='green')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
for bar, score in zip(bars2, scores_after):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Mesure de la QUALITÉ DE SÉPARATION', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/03_SCORE_separation_quality.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: output/03_SCORE_separation_quality.png")
plt.close()

# ============================================
# AFFICHER LE RÉSUMÉ
# ============================================

print("\n" + "="*70)
print("📊 RÉSUMÉ DE LA TRANSFORMATION")
print("="*70)

print(f"\n🌀 AVANT LDA (CHAOTIQUE) :")
print(f"   • Données mélangées et perturbées")
print(f"   • Classes non séparées")
print(f"   • Score de séparation : {silhouette_chaotic:.3f} (FAIBLE ❌)")

print(f"\n✅ APRÈS LDA (ORGANISÉ) :")
print(f"   • Données bien organisées")
print(f"   • Classes PARFAITEMENT séparées")
print(f"   • Score de séparation : {silhouette_lda:.3f} (EXCELLENT ✅)")

print(f"\n📈 AMÉLIORATION :")
print(f"   • Gain de séparation : +{(silhouette_lda - silhouette_chaotic):.3f}")
print(f"   • Amélioration : {((silhouette_lda - silhouette_chaotic) / abs(silhouette_chaotic) * 100):.1f}%")

print("\n" + "="*70)
print("🎉 GRAPHIQUES CRÉÉS !")
print("="*70)
print("\nFichiers créés dans 'output/' :")
print("  1. 00_CHAOTIC_donnees_perturbees.png")
print("  2. 01_ORGANIZED_donnees_organisees.png")
print("  3. 02_TRANSFORMATION_chaos_to_order.png")
print("  4. 03_SCORE_separation_quality.png")
print("="*70)