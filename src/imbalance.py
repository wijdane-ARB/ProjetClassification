"""
src/imbalance.py
Pipeline de traitement du déséquilibre — Binôme B

Usage:
    from src.imbalance import apply_smote, apply_adasyn, get_class_weight

Règle: appliquer UNIQUEMENT sur X_train — jamais sur val ou test.
"""

from imblearn.over_sampling  import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.combine        import SMOTETomek
import numpy as np

RANDOM_SEED = 42


def apply_smote(X_train, y_train, seed=RANDOM_SEED):
    """
    Oversampling SMOTE — génère des exemples synthétiques
    par interpolation entre k voisins les plus proches.
    Résultat : classes équilibrées 1:1.
    """
    return SMOTE(random_state=seed).fit_resample(X_train, y_train)


def apply_adasyn(X_train, y_train, seed=RANDOM_SEED):
    """
    Oversampling ADASYN — comme SMOTE mais concentre
    la génération dans les zones difficiles à classifier.
    """
    return ADASYN(random_state=seed).fit_resample(X_train, y_train)


def apply_nearmiss(X_train, y_train, version=1):
    """
    Undersampling NearMiss — réduit la classe majoritaire.
    version=1 : garde les majoritaires proches des minoritaires.
    version=2 : garde les majoritaires loin des minoritaires.
    ⚠️  Perte d'information — à utiliser avec précaution.
    """
    return NearMiss(version=version).fit_resample(X_train, y_train)


def apply_smote_tomek(X_train, y_train, seed=RANDOM_SEED):
    """
    Approche combinée — SMOTE + suppression des Tomek links.
    Surcrée des exemples minoritaires et nettoie la frontière.
    """
    return SMOTETomek(random_state=seed).fit_resample(X_train, y_train)


def get_class_weight(y_train):
    """
    Retourne class_weight='balanced' pour sklearn.
    Usage: LogisticRegression(class_weight=get_class_weight(y_train))
           RandomForestClassifier(class_weight=get_class_weight(y_train))
    """
    return 'balanced'


def get_scale_pos_weight(y_train):
    """
    Retourne scale_pos_weight pour XGBoost.
    Formule : n_négatifs / n_positifs
    Usage: XGBClassifier(scale_pos_weight=get_scale_pos_weight(y_train))
    """
    counts = np.bincount(y_train)
    return counts[0] / counts[1]