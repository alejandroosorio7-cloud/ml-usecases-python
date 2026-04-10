import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor

def generar_caso_de_uso_crear_features_temporales():
    n = random.randint(20, 50)

    fechas = pd.date_range(start="2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "fecha": fechas,
        "valor": np.random.randn(n)
    })

    input_data = {"df": df.copy(), "target_col": "valor"}

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["mes"] = df["fecha"].dt.month
    df["fin_semana"] = (df["dia_semana"] >= 5).astype(int)

    X = df[["dia_semana", "mes", "fin_semana"]]
    y = df["valor"]

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    return input_data, model.feature_importances_
