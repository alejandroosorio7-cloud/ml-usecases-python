import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def generar_caso_de_uso_evaluar_con_stratified_kfold():
    n = random.randint(30, 60)
    m = random.randint(2, 5)

    df = pd.DataFrame(np.random.randn(n, m), columns=[f"x{i}" for i in range(m)])
    df["target"] = np.random.choice([0, 1], size=n, p=[0.7, 0.3])

    input_data = {"df": df.copy(), "target_col": "target"}

    X = df.drop(columns=["target"]).values
    y = df["target"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in skf.split(X, y):
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        scores.append(accuracy_score(y[test_idx], y_pred))

    return input_data, (np.mean(scores), np.std(scores))
