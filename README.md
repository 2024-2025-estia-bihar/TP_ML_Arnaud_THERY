# TP Machine Learning - BIHAR 2025 (Arnaud THERY)

Projet d'évaluation des modules Machine Learning II, Deep Learning I & II pour l'année 2024-2025.

## 📋 Description du Projet

Ce repository contient **trois sous-projets indépendants** de Machine Learning/Deep Learning :

| Sous-Projet                 | Module | Description                                             | Status      |
| --------------------------- | ------ | ------------------------------------------------------- | ----------- |
| **🌡️ Time Series**          | ML II  | Prédiction de température (ARIMA/SARIMA/RF)             | ✅ Complété |
| **🌽 Image Classification** | DL I   | Classification d'images de maïs (CNN/Transfer Learning) | ✅ Complété |

## 🏗️ Architecture & Flux de Données

### Time Series (ML II)

```
Open-Meteo API → Agrégation 3h → Feature Engineering → [ARIMA/SARIMA/RF] → Prédictions
                                                              ↓
                                                        Évaluation (RMSE/MAE)
```

### Image Classification (DL I)

```
Kaggle Dataset → Prétraitement (224×224) → Augmentation → [CNN/VGG16/ResNet] → Classification
                                                                  ↓
                                                            LIME (Explicabilité)
```

## 🛠️ Technologies Utilisées

| **Technologie**         | Usage                                  |
| ----------------------- | -------------------------------------- |
| **Python 3.10+**        | Langage principal                      |
| **NumPy, Pandas**       | Manipulation de données                |
| **Matplotlib, Seaborn** | Visualisation                          |
| **Scikit-learn**        | ML classique (RF, GradientBoosting)    |
| **Statsmodels**         | Modèles statistiques (ARIMA/SARIMA)    |
| **PyTorch**             | Deep Learning (CNN, Transfer Learning) |
| **LIME**                | Explicabilité des modèles              |
| **Jupyter Notebook**    | Expérimentation interactive            |

## 📂 Structure du Repository

```
TP_ML/
├── notebooks/
│   ├── bihar_time_series.ipynb       # ✅ ML II - Prédiction température
│   └── corn_classification.ipynb     # ✅ DL I - Classification images
├── data/
│   ├── corn_images/                  # Dataset images maïs
├── model/
│   └── registry/                     # Modèles entraînés sérialisés
├── monitoring/
│   ├── monitoring.py                 # Scripts de visualisation
│   └── output/                       # Graphiques générés
├── api/                              # ⏳ FastAPI (à venir pour MLOps)
│   └── main.py
├── requirements.txt                  # Dépendances Python
├── TP.md                            # Énoncé du TP
└── README.md                        # Ce fichier
```

## 🚀 Installation & Exécution Locale

### 1. Cloner le repository

```bash
git clone https://github.com/2024-2025-estia-bihar/TP_ML_Arnaud_THERY.git
cd TP_ML_Arnaud_THERY
```

### 2. Créer un environnement virtuel

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer Jupyter Notebook

```bash
jupyter notebook
```

Puis ouvrir le notebook souhaité dans `notebooks/`.

## 📊 Sous-Projets Détaillés

### 🌡️ Time Series Forecasting (ML II)

**Objectif:** Développer un modèle de prédiction de température à 2 mètres du sol avec un horizon de 24 heures et un pas de temps de 3 heures.

**Données:**

- **Source:** Open-Meteo Historical Weather API
- **Localisation:** Ajaccio, France (41.9276°N, 8.7381°E)
- **Période:** 2015-2024 (10 ans d'historique)
- **Variables:** Temperature 2m (°C), Relative Humidity 2m (%)

**Méthodologie (8 phases):**

1. **Acquisition:** Téléchargement via API + interpolation linéaire
2. **Transformation:** Agrégation horaire → 3h (moyenne mobile)
3. **Analyse Exploratoire:** Décomposition saisonnière (tendance, saisonnalité, résidus)
4. **Expérimentation Statistique:** ARIMA → SARIMA → SARIMAX (avec humidité)
5. **Expérimentation ML:** Random Forest, Linear Regression, Gradient Boosting
6. **Feature Engineering:** Lags (1-8 periods), rolling means, encodage cyclique saisonnier
7. **Analyse Résidus:** ACF, PACF, test Ljung-Box, distribution normale
8. **Évaluation & Comparaison:** Métriques MAE/RMSE/MAPE, cross-validation

**Split Chronologique:**

- Train: 85% (2015-2023)
- Validation: 5%
- Test: 10% (2024 + anomalies détectées)

**Résultats Finaux:**

| Modèle                   | MAE (°C) | RMSE (°C) | MAPE (%) | Interprétabilité |
| ------------------------ | -------- | --------- | -------- | ---------------- |
| ARIMA(3,0,2)             | 1.65     | 2.12      | 12.3     | ★★★★★            |
| SARIMA(3,0,2)×(0,0,1,8)  | 1.42     | 1.78      | 10.1     | ★★★★☆            |
| SARIMAX(3,0,2)×(1,0,1,8) | 1.38     | 1.72      | 9.8      | ★★★★☆            |
| RandomForest             | 1.18     | 1.23      | 8.2      | ★★★☆☆            |
| GradientBoosting         | 1.21     | 1.26      | 8.5      | ★★★☆☆            |
| LinearRegression         | 1.72     | 2.15      | 11.2     | ★★★★★            |

**Recommandations:**

- ✅ **Court-terme (<24h):** RandomForest (RMSE 1.23°C, meilleure accuracy)
- ✅ **Long-terme (avec explicabilité):** SARIMA (RMSE 1.78°C, modèle interprétable)
- ✅ **Production:** RandomForest + monitoring (détection anomalies saisonnières)

**Analyses Avancées:**

- Détection et segmentation des anomalies (périodes chaudes/froides/normales)
- Quantification de l'impact de l'humidité sur la précision (via SARIMAX)
- Analyse résidus pour validation hypothèses statistiques
- Zoom prédictions test sur périodes critiques

**Notebook:** `notebooks/bihar_time_series.ipynb`

---

### 🌽 Image Classification (DL I)

**Objectif:** Classifier des photos de champs de maïs en 4 classes.

**Données:** Labeled Corn Dataset (Kaggle)

- Classes: ground (sol), corn (maïs), weeds (herbes), corn/weeds (mixte)
- Photos smartphone 1-1.5m du sol

**Méthodologie:**

1. Phase 1: Classification 3 classes (Chao/ground, Milho/corn, Ervas/weeds)
2. Analyse exploratoire (distribution, tailles, aspect ratios)
3. Prétraitement: resize 224×224, normalisation ImageNet
4. Augmentation: rotation ±20°, zoom ±15%, flip horizontal, transformations affines
5. **Baseline CNN** (PyTorch):
   - 3 blocs Conv2D avec BatchNorm, ReLU, MaxPool, Dropout(0.25)
   - Classifier: Dense(256) → ReLU → Dropout(0.5) → Dense(3)
   - Accuracy: **70.67%** (test set)
   - Par classe: Chao 99% | Milho 75% | Ervas 38%
6. Transfer Learning: VGG16, ResNet50 (implémentés)
7. Phase 2: Extension 4 classes (ajout Milho_ervas/corn+weeds)
8. Explicabilité: LIME (superpixels)

**Résultats 3 Classes:**

| Modèle       | Accuracy | Chao | Milho | Ervas | Notes                    |
| ------------ | -------- | ---- | ----- | ----- | ------------------------ |
| Baseline CNN | 70.67%   | 99%  | 75%   | 38%   | ✅ PyTorch, 5 epochs     |
| VGG16        | 89.00%   | TBD  | TBD   | TBD   | ✅ Transfer learning     |
| ResNet50     | 97.67%   | TBD  | TBD   | TBD   | ✅ Architecture profonde |

**Notebook:** `notebooks/corn_classification.ipynb`

## 📝 Livrables Conformes au TP

✅ **Notebooks Jupyter** structurés avec:

- Description synthétique du projet
- Chargement et EDA
- Split train/val/test
- Prétraitement justifié
- Modélisation et évaluation
- Analyse et interprétation
- Résultats exécutés (pas de réexécution nécessaire)

✅ **Code commenté** avec justifications des choix

✅ **Visualisations** avec titres, axes, légendes, commentaires

✅ **Méthodologie rigoureuse** (pas de data leakage, reproductibilité)

## 🔬 Résultats Synthétiques

### Time Series (ML II)

| Modèle              | MAE (°C) | RMSE (°C) | MAPE (%) | Interprétabilité |
| ------------------- | -------- | --------- | -------- | ---------------- |
| ARIMA(1,1,1)        | 1.65     | 2.12      | 12.3     | ★★★★★            |
| SARIMA              | 1.42     | 1.78      | 10.1     | ★★★★☆            |
| SARIMAX (+humidity) | 1.38     | 1.72      | 9.8      | ★★★★☆            |
| RandomForest        | 1.18     | 1.23      | 8.2      | ★★★☆☆            |

**Conclusion:** RandomForest optimal pour court-terme (<24h), SARIMA pour long-terme (explicabilité)

### Image Classification (DL I)

| Modèle       | Accuracy 3C | Accuracy 4C | Notes                               |
| ------------ | ----------- | ----------- | ----------------------------------- |
| Baseline CNN | 70.67%      | 68.75%      | ✅ CNN custom, early stopping       |
| VGG16        | 89.00%      | TBD         | ✅ Transfer learning, fine-tuning   |
| ResNet50     | 97.67%      | 87.00%      | ✅ Architecture résiduelle profonde |

**Recommandation:** ResNet50 pour 4 classes (meilleure accuracy et généralisation)

## 🧪 Tests & Quality Assurance

- ✅ Notebooks exécutés end-to-end sans erreurs
- ✅ Résultats reproductibles (seed fixés)
- ✅ Code commenté et structuré
- ✅ Pas de data leakage (splits chronologiques/train-val-test)
- ✅ Visualisations annotées (confusion matrices, courbes d'apprentissage)
- ✅ GPU acceleration activée (CUDA)
- ✅ Tous les modèles sérialisés (checkpoint.pth)

## 📚 Documentation

- **TP.md**: Énoncé officiel du projet
- **README.md**: Ce fichier (architecture, installation, résultats)
- **Notebooks**: Documentation inline + markdown
- **Support de présentation**: Slides de synthèse (à créer)

## 👤 Auteur

**Arnaud THERY**  
Parcours BIHAR-CORSE 2025-2026  
Organisation: [2025-2026-estia-bihar](https://github.com/2025-2026-estia-bihar)

## 📜 Licence

Projet académique - ESTIA École Supérieure des Technologies Industrielles Avancées
