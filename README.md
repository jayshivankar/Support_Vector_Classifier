# 🎤 Voice Gender Classification using SVC

This repository contains a machine learning pipeline to classify gender based on voice features using a **Support Vector Classifier (SVC)**. The dataset used is `voice.csv`, which contains acoustic properties of male and female voices.

## 📁 Dataset

The dataset consists of various voice features such as:
- `meanfreq`, `sd`, `median`, `Q25`, `Q75`, `IQR`, `skew`, `kurt`, `sp.ent`, `sfm`, `mode`, `centroid`, `meanfun`, `minfun`, `maxfun`, `meandom`, `mindom`, `maxdom`, `dfrange`, `modindx`
- `label`: `male` or `female`

## 📊 Objective

To classify voices into **male** or **female** using **Support Vector Machines**, with feature scaling and **hyperparameter tuning** using `GridSearchCV`.

## ⚙️ Preprocessing Steps

- Loaded the dataset using pandas
- Label encoded the `label` column (`male` → 0, `female` → 1)
- Split the data into training and test sets
- Applied `StandardScaler` to normalize features

## 🧠 Model: Support Vector Classifier (SVC)

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
```

## 🔍 Hyperparameter Tuning

GridSearchCV was used to find the best combination of parameters:

```python
estimator = SVC(random_state=42)

param_grid = {
    'kernel': ['rbf', 'poly'],
    'C': np.arange(0, 5),
    'gamma': np.arange(0, 2)
}

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=4
)

grid_search.fit(x_train_scaled, y_train)

best_parameters = grid_search.best_params_
print(best_parameters)
```

## ✅ Best Parameters Found

Example output:
```python
{'C': 3, 'gamma': 1, 'kernel': 'rbf'}
```

## 📈 Results

- Evaluated the final model on the test set using:
  - Accuracy
  - Classification Report
  - Confusion Matrix

## 🧪 Requirements

```bash
pip install pandas numpy scikit-learn
```

## ▶️ Run the Code

```bash
python svc_voice_classifier.py
```
