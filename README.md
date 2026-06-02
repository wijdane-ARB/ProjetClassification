<div align="center">

<br/>

#  Détection de Fraude par Carte Bancaire
### *Credit Card Fraud Detection — Pipeline ML Complet*

<br/>

[![Python](https://img.shields.io/badge/Python-3.14%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optimisation%20Bayésienne-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Interprétabilité-8A2BE2?style=for-the-badge)](https://shap.readthedocs.io/)

<br/>

> *Détection de transactions frauduleuses sur un dataset bancaire fortement déséquilibré (ratio 578:1) — de l'EDA à l'interprétabilité SHAP, en passant par la calibration des probabilités.*

<br/>

</div>

---

##  Contexte & Problématique

Ce projet traite un problème de **classification binaire fortement déséquilibrée** : détecter les transactions frauduleuses parmi des millions d'opérations légitimes.

Le dataset [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) contient **284 807 transactions** réalisées sur 2 jours, dont seulement **492 fraudes (0.172%)** — soit un ratio de **578:1** entre légitimes et frauduleuses. Les variables V1–V28 sont des composantes issues d'une **ACP (PCA)** pour des raisons de confidentialité.

**Enjeu principal :** avec un tel déséquilibre, un modèle naïf qui prédit "tout légitime" atteint 99.8% d'accuracy — ce qui est parfaitement inutile. Le projet utilise donc des métriques adaptées (F1-Macro, AUPRC, MCC) et des techniques spécifiques pour traiter ce déséquilibre.

---

##  Structure du Projet

```
ProjetClassification/
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploration & Feature Engineering
│   ├── 02_imbalance.ipynb            # Traitement du déséquilibre
│   ├── 03_logistic.ipynb             # Régression Logistique Elastic Net
│   ├── 04_random_forest.ipynb        # Random Forest + Matrice de Proximité
│   ├── 05_xgboost.ipynb              # XGBoost + Optimisation Bayésienne (Optuna)
│   └── 06_calibration_shap_v2.ipynb  # Calibration & Interprétabilité SHAP
│
├── src/
│   ├── imbalance.py                  # Fonctions de rééchantillonnage réutilisables
│   └── evaluate.py                   # Fonction d'évaluation partagée (F1, AUPRC, MCC)
│
├── data/
│   ├── raw/creditcard.csv            # Dataset brut Kaggle
│   └── processed/                    # Splits train/val/test (produits par notebook 01)
│
├── models/                           # Modèles entraînés (.pkl)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
│
├── plots/                            # Visualisations générées (~30 graphiques)
├── main.py
└── pyproject.toml
```

---

##  Démarche & Étapes

Le projet est organisé en **6 notebooks séquentiels**, chaque étape s'appuyant sur la précédente.

---

### `01` — Analyse Exploratoire & Feature Engineering

**Objectif :** Comprendre les données, créer des features pertinentes, produire les splits propres.

- **Exploration de base** : 284 807 lignes × 31 colonnes, aucune valeur manquante. Ratio fraudes/légitimes : 492 / 284 315 (~578:1).
- **Feature Engineering** :
  - `Amount_log` : transformation `log1p` du montant (distribution très asymétrique, max = 25 691€)
  - `Hour` : extraction de l'heure réelle depuis `Time` (secondes cumulées → `% 86400 // 3600`)
  - Suppression des colonnes brutes `Time` et `Amount`
- **Analyse de colinéarité** : matrice de corrélation + VIF sur toutes les features. Les composantes V1–V28 sont orthogonales par construction PCA → aucune multicolinéarité sévère (VIF < 10 pour toutes).
- **Splits produits** : `train.csv` (70%), `val.csv` (10%), `test.csv` (20%) — stratifiés, avec `random_state=42`.

---

### `02` — Traitement du Déséquilibre

**Objectif :** Identifier la meilleure stratégie pour gérer le ratio 578:1.

Deux familles d'approches sont comparées avec une Régression Logistique comme modèle de référence :

| Approche | Méthode | Principe |
|---|---|---|
| **Algorithmique** | `class_weight='balanced'` | Pénalise davantage les erreurs sur la classe minoritaire (~289× plus pour les fraudes) |
| **Algorithmique** | `class_weight` manuel `{0:1, 1:578}` | Ratio exact négatifs/positifs |
| **Données** | SMOTE | Génère des exemples synthétiques par interpolation entre voisins proches |
| **Données** | ADASYN | Comme SMOTE, mais se concentre sur les zones de décision difficiles |
| **Données** | NearMiss v1 | Sous-échantillonnage — garde les majoritaires proches des minoritaires |
| **Données** | SMOTETomek | Combiné — SMOTE + suppression des exemples ambigus (Tomek links) |

>  **Règle absolue respectée :** SMOTE/ADASYN/NearMiss appliqués **uniquement** sur `X_train`. Jamais sur val ni test.

**Métriques utilisées :** F1-Macro, AUPRC, MCC, Rappel, Précision. L'accuracy est volontairement exclue (non informative sur données déséquilibrées).

---

### `03` — Régression Logistique avec Pénalité Elastic Net

**Objectif :** Entraîner un modèle linéaire interprétable et optimisé.

**Choix de l'Elastic Net :** combinaison L1 (Lasso) + L2 (Ridge).

$$\text{Loss} = -\log(L) + C^{-1} \left[ \alpha \|w\|_1 + (1-\alpha) \|w\|_2^2 \right]$$

- L1 effectue une sélection automatique des features (V1–V28 potentiellement redondantes)
- L2 stabilise l'estimation sur features corrélées
- Solver `saga` (requis pour `elasticnet`)

**Optimisation :** `RandomizedSearchCV` (20 itérations × 3 folds stratifiés) — ~8 à 12 minutes.
- Espace de recherche continu : `C ∈ loguniform(0.001, 10)`, `l1_ratio ∈ uniform(0.05, 0.95)`
- Scoring : `average_precision` (AUPRC)
- `class_weight='balanced'`

**Résultats complémentaires :**
- Visualisation des **20 coefficients les plus influents**
- Optimisation du **seuil de décision** sur val (sans data leakage)
- Courbes **Precision-Recall** et **ROC**
- Courbe de calibration initiale

---

### `04` — Random Forest avec Matrice de Proximité

**Objectif :** Modèle ensembliste + analyse structurale via la proximité inter-observations.

**Optimisation :** `RandomizedSearchCV` (20 itérations × 5 folds), espace de recherche :
- `n_estimators` ∈ {100, 200, 300}
- `max_depth` ∈ {10, 20, 30, None}
- `min_samples_split` ∈ {2, 5, 10}
- `min_samples_leaf` ∈ {1, 2, 4}
- `max_features` ∈ {`sqrt`, `log2`}

**Matrice de Proximité :**

$$\text{Proximité}(i,j) = \frac{\text{Nombre d'arbres où } i \text{ et } j \text{ sont dans la même feuille terminale}}{\text{Nombre total d'arbres}}$$

Calculée sur un **échantillon stratifié de 5 000 observations** (la matrice NxN avec N=205 000 nécessiterait ~336 GB de RAM). Utilisée pour :
- Visualiser la séparation fraudes/légitimes dans l'espace de proximité (PCA 2D sur matrice de distance)
- Détecter les **outliers de prédiction** — transactions légitimes classées comme fraudes et vice-versa

---

### `05` — XGBoost avec Optimisation Bayésienne (Optuna)

**Objectif :** Modèle de gradient boosting avec recherche d'hyperparamètres avancée et deux stratégies cost-sensitive.

**Principe XGBoost :** chaque arbre $f_t$ corrige les résidus (gradients) du modèle précédent :

$$\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta \cdot f_t(x)$$

**Deux stratégies pour le déséquilibre :**

| Stratégie | Mécanisme | Paramètre clé |
|---|---|---|
| `scale_pos_weight` | Multiplie le gradient de la classe positive par `neg/pos ≈ 578` | `scale_pos_weight = 578` |
| **Focal Loss** (Lin et al., 2017) | Réduit la contribution des exemples faciles, focalise sur les difficiles | `α=0.75, γ=2` |

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

**Optimisation Bayésienne via Optuna (TPE) — 50 essais :**

| Hyperparamètre | Espace de recherche | Justification |
|---|---|---|
| `max_depth` | [3, 10] | < 3 = underfitting, > 10 = overfitting |
| `learning_rate` η | loguniform[0.01, 0.3] | Petit η = meilleure généralisation |
| `n_estimators` | [100, 500] | Compensé par η |
| `subsample` | [0.6, 1.0] | Réduction de la variance |
| `colsample_bytree` | [0.6, 1.0] | Diversité des arbres |
| `lambda` (L2) | loguniform[0.001, 10] | Régularisation Ridge |
| `alpha` (L1) | loguniform[0.001, 10] | Sélection de features |
| `min_child_weight` | [1, 10] | Régularisation des feuilles |
| `gamma` | [0, 5] | Élagage — gain minimum pour diviser |

---

### `06` — Calibration & Interprétabilité SHAP

**Objectif :** S'assurer que les probabilités prédites sont fiables, puis comprendre *pourquoi* le modèle prédit une fraude.

#### Calibration

Un modèle bien calibré doit vérifier : parmi les transactions auxquelles il attribue 70% de probabilité de fraude, 70% doivent effectivement être des fraudes.

**Mesures :** ECE (Expected Calibration Error) + Brier Score sur les 3 modèles.

**Méthodes testées :**
- **Platt Scaling** : régression logistique sur les scores bruts (adapté aux calibrations monotones)
- **Isotonic Regression** : non-paramétrique, adapté aux calibrations non-monotones

**Résultats clés :**
- LR → Platt Scaling (ECE : 0.47 → 0.10)
- RF → Isotonic obligatoire (Platt empire : ECE 0.12 → 0.38)
- XGB → déterminé par comparaison ECE

#### Interprétabilité SHAP

**SHAP (SHapley Additive exPlanations)** décompose chaque prédiction en contributions individuelles des features :

- **Analyse globale** : quelles features influencent le plus les prédictions de fraude en moyenne
- **Analyse locale** : pour une transaction suspecte donnée, quelles features ont poussé le modèle vers "fraude"
- **Synthèse** des variables les plus déterminantes sur l'ensemble du test set

---

##  Métriques & Évaluation

L'accuracy est **volontairement exclue** de toutes les évaluations (trompeuse sur données déséquilibrées).

| Métrique | Description | Pourquoi l'utiliser |
|---|---|---|
| **F1-Macro** | Moyenne harmonique precision/recall sur toutes les classes | Équilibre entre classes |
| **AUPRC** | Aire sous la courbe Precision-Recall | Robuste au déséquilibre — plus informatif que l'AUC-ROC |
| **MCC** | Matthews Correlation Coefficient ∈ [-1, 1] | Prend en compte TP, TN, FP, FN simultanément |
| **F1-Fraude** | F1 sur la classe minoritaire uniquement | Mesure directe de la détection |
| **Rappel** | Fraction des fraudes réelles détectées | Priorité : ne pas manquer une fraude |

---

##  Installation & Lancement

### Prérequis

- Python **3.14** ou supérieur
- pip

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/wijdane-ARB/ProjetClassification.git
cd ProjetClassification

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -e .

# 4. Télécharger le dataset
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# → placer creditcard.csv dans data/raw/

# 5. Exécuter les notebooks dans l'ordre
jupyter notebook notebooks/
```

### Dépendances principales

```
pandas · numpy · scikit-learn · imbalanced-learn
xgboost · optuna · shap
matplotlib · seaborn · joblib
statsmodels (VIF)
```

---

##  Stack Technique

| Domaine | Outils |
|---|---|
| **Langage & Notebooks** | Python 3.14+, Jupyter Notebook |
| **Manipulation de données** | pandas, numpy |
| **Machine Learning** | scikit-learn, xgboost |
| **Déséquilibre** | imbalanced-learn (SMOTE, ADASYN, NearMiss, SMOTETomek) |
| **Optimisation** | Optuna (TPE — Tree-structured Parzen Estimator) |
| **Interprétabilité** | SHAP |
| **Visualisation** | matplotlib, seaborn |
| **Sérialisation** | joblib |
| **Statistiques** | statsmodels (VIF) |

---

##  Équipe

Ce projet a été réalisé dans le cadre d'un travail académique par :

<br/>

<div align="center">

| 👤 Membre | Contribution principale |
|-----------|------------------------|
| **AIT EL MAHJOUB Abdessamad** | EDA, Feature Engineering, Analyse de colinéarité |
| **TOKO Rayane** | Traitement du déséquilibre (SMOTE, ADASYN, NearMiss, SMOTETomek) |
| **AAROUB Wijdane** | XGBoost, Optimisation Bayésienne (Optuna), Focal Loss |
| **BOUSSAID Mohamed Amine** | Calibration (Platt / Isotonic), Interprétabilité SHAP |

</div>

<br/>

---

<div align="center">

*2025 / 2026*

</div>
