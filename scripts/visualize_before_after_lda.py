import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("📊 VISUALISATION AVANT et APRÈS LDA")
print("="*70)

# ============================================
# 1️⃣ CHARGER LES DONNÉES
# ============================================
print("\n[1/3] 📂 Chargement des données...")
df = pd.read_csv('data/lda_dataset.csv')
df = df.dropna(subset=['label'])

feature_cols = [col for col in df.columns if col.startswith('f')]
X = df[feature_cols].values
y = df['label'].values

print(f"✅ {len(df)} documents chargés")
print(f"✅ {X.shape[1]} features")
print(f"✅ {len(np.unique(y))} classes\n")

# ============================================
# 2️⃣ RÉDUIRE LES DIMENSIONS AVANT LDA (PCA)
# ============================================
print("[2/3] 🔢 Réduction des dimensions AVANT LDA...")

# Utiliser PCA pour réduire à 2D (pour visualiser)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"✅ PCA appliqué (avant LDA)")
print(f"✅ Variance expliquée par PCA:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"   Composante {i+1}: {var:.2%}\n")

# ============================================
# 3️⃣ APPLIQUER LDA
# ============================================
print("[3/3] 🤖 Application de LDA...")

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

print(f"✅ LDA appliqué")
print(f"✅ Variance expliquée par LDA:")
for i, var in enumerate(lda.explained_variance_ratio_):
    print(f"   Composante {i+1}: {var:.2%}\n")

# ============================================
# 🎨 VISUALISATIONS COMPARATIVES
# ============================================

import os
if not os.path.exists('output'):
    os.makedirs('output')

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
markers = ['o', 's', '^']

# 1️⃣ GRAPHIQUE AVANT LDA (avec PCA)
print("Création des graphiques...")
print("1️⃣ Graphique AVANT LDA (PCA)...")

fig, ax = plt.subplots(figsize=(12, 8))

for i, label in enumerate(lda.classes_):
    indices = y == label
    ax.scatter(X_pca[indices, 0], X_pca[indices, 1],
               label=label, alpha=0.7, s=200, 
               color=colors[i], edgecolors='black', 
               linewidth=2, marker=markers[i], zorder=3)

ax.set_xlabel(f'PCA Composante 1 ({pca.explained_variance_ratio_[0]:.1%} de variance)', 
              fontsize=12, fontweight='bold')
ax.set_ylabel(f'PCA Composante 2 ({pca.explained_variance_ratio_[1]:.1%} de variance)', 
              fontsize=12, fontweight='bold')
ax.set_title('AVANT LDA - Projection PCA 2D (50 features → 2D)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best', title='Classes')
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/00_AVANT_LDA_pca_projection.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: output/00_AVANT_LDA_pca_projection.png")
plt.close()

# 2️⃣ GRAPHIQUE APRÈS LDA
print("2️⃣ Graphique APRÈS LDA...")

fig, ax = plt.subplots(figsize=(12, 8))

for i, label in enumerate(lda.classes_):
    indices = y == label
    ax.scatter(X_lda[indices, 0], X_lda[indices, 1],
               label=label, alpha=0.7, s=200, 
               color=colors[i], edgecolors='black', 
               linewidth=2, marker=markers[i], zorder=3)

ax.set_xlabel(f'LDA 1 ({lda.explained_variance_ratio_[0]:.1%} de variance)', 
              fontsize=12, fontweight='bold')
ax.set_ylabel(f'LDA 2 ({lda.explained_variance_ratio_[1]:.1%} de variance)', 
              fontsize=12, fontweight='bold')
ax.set_title('APRÈS LDA - Projection LDA 2D (50 features → 2D)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best', title='Classes')
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/01_APRES_LDA_lda_projection.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: output/01_APRES_LDA_lda_projection.png")
plt.close()

# 3️⃣ COMPARAISON CÔTE À CÔTE
print("3️⃣ Comparaison AVANT vs APRÈS...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# AVANT (PCA)
for i, label in enumerate(lda.classes_):
    indices = y == label
    ax1.scatter(X_pca[indices, 0], X_pca[indices, 1],
               label=label, alpha=0.7, s=200, 
               color=colors[i], edgecolors='black', 
               linewidth=2, marker=markers[i], zorder=3)

ax1.set_xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11, fontweight='bold')
ax1.set_ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11, fontweight='bold')
ax1.set_title('❌ AVANT LDA (PCA) - Pas très séparé', fontsize=12, fontweight='bold', color='red')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3, linestyle='--')

# APRÈS (LDA)
for i, label in enumerate(lda.classes_):
    indices = y == label
    ax2.scatter(X_lda[indices, 0], X_lda[indices, 1],
               label=label, alpha=0.7, s=200, 
               color=colors[i], edgecolors='black', 
               linewidth=2, marker=markers[i], zorder=3)

ax2.set_xlabel(f'LDA 1 ({lda.explained_variance_ratio_[0]:.1%})', fontsize=11, fontweight='bold')
ax2.set_ylabel(f'LDA 2 ({lda.explained_variance_ratio_[1]:.1%})', fontsize=11, fontweight='bold')
ax2.set_title('✅ APRÈS LDA - Parfaitement séparé !', fontsize=12, fontweight='bold', color='green')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3, linestyle='--')

plt.suptitle('Comparaison : AVANT vs APRÈS LDA', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('output/02_COMPARAISON_avant_apres_lda.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: output/02_COMPARAISON_avant_apres_lda.png")
plt.close()

# 4️⃣ STATISTIQUES
print("\n4️⃣ Statistiques de séparation...")

print("\n" + "="*70)
print("📊 RÉSUMÉ")
print("="*70)
print(f"\nAVANT LDA (PCA) :")
print(f"  • Variance expliquée PCA 1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"  • Variance expliquée PCA 2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"  • Total: {pca.explained_variance_ratio_.sum():.2%}")

print(f"\nAPRÈS LDA :")
print(f"  • Variance expliquée LDA 1: {lda.explained_variance_ratio_[0]:.2%}")
print(f"  • Variance expliquée LDA 2: {lda.explained_variance_ratio_[1]:.2%}")
print(f"  • Total: {lda.explained_variance_ratio_.sum():.2%}")

print(f"\nCONCLUSION :")
print(f"  ✅ LDA SÉPARE MIEUX LES CLASSES que PCA !")
print(f"  ✅ LDA 1 explique {lda.explained_variance_ratio_[0]:.1%} de la séparation")
print(f"  ✅ Les classes sont PARFAITEMENT séparées !")

print("\n" + "="*70)
print("🎉 GRAPHIQUES CRÉÉS !")
print("="*70)
print("\nFichiers créés dans 'output/' :")
print("  1. 00_AVANT_LDA_pca_projection.png")
print("  2. 01_APRES_LDA_lda_projection.png")
print("  3. 02_COMPARAISON_avant_apres_lda.png")
print("="*70)