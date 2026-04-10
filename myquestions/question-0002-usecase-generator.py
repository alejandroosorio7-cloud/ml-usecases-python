import pandas as pd
import numpy as np
import random
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def generar_caso_de_uso_pipeline_preprocesamiento_modelo():
    n = random.randint(20, 50)

    df = pd.DataFrame({
        "num1": np.random.randn(n),
        "num2": np.random.randn(n),
        "cat": np.random.choice(["A", "B", "C"], n),
        "target": np.random.randint(0, 2, n)
    })

    input_data = {"df": df.copy(), "target_col": "target"}

    X = df.drop(columns=["target"])
    y = df["target"]

    num_cols = ["num1", "num2"]
    cat_cols = ["cat"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(), cat_cols)
    ])

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=200))
    ])

    pipeline.fit(X, y)

    return input_data, pipeline
